//===-- Mangled.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Mangled.h"

#include "lldb/Core/DataFileCache.h"
#include "lldb/Core/DemangledNameInfo.h"
#include "lldb/Target/Language.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-enumerations.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

#include <array>
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

using namespace lldb_private;

#pragma mark Mangled

namespace {
/// A mangling scheme that a language plugin registered.
struct RegisteredManglingScheme {
  static constexpr size_t g_inline_prefixes = 12;
  static constexpr size_t g_inline_prefix_length = 16;

  llvm::SmallVector<llvm::SmallString<g_inline_prefix_length>,
                    g_inline_prefixes>
      prefixes;
  /// If set, a name that starts with one of the prefixes must also match.
  Mangled::ManglingSchemeMatcher matches = nullptr;
  lldb::LanguageType language = lldb::eLanguageTypeUnknown;
  /// Schemes are never removed, so that GetManglingScheme() can read them
  /// without a lock. Unregistered schemes are only marked inactive. No other
  /// data is published through this flag, so it needs no memory ordering.
  std::atomic<bool> active = false;
};

/// The mangling schemes that language plugins registered.
class ManglingSchemeRegistry {
public:
  Mangled::ManglingScheme Register(llvm::ArrayRef<llvm::StringRef> prefixes,
                                   lldb::LanguageType language,
                                   Mangled::ManglingSchemeMatcher matches);
  void Unregister(Mangled::ManglingScheme scheme);
  lldb::LanguageType GetLanguage(Mangled::ManglingScheme scheme) const;
  Mangled::ManglingScheme Find(llvm::StringRef name) const;

private:
  /// Serializes registering and unregistering schemes.
  std::mutex m_mutex;
  std::array<RegisteredManglingScheme,
             Mangled::g_num_mangling_schemes -
                 Mangled::eFirstRegisteredManglingScheme>
      m_schemes;
  /// The number of initialized entries in m_schemes. Incrementing it publishes
  /// a new entry to readers.
  std::atomic<size_t> m_num_schemes = 0;
};
} // namespace

static ManglingSchemeRegistry &GetManglingSchemeRegistry() {
  // Never destroyed, since other threads may still look up mangling schemes
  // while static destructors run at exit.
  static ManglingSchemeRegistry *g_registry = new ManglingSchemeRegistry();
  return *g_registry;
}

static Mangled::ManglingScheme SchemeForIndex(size_t index) {
  return static_cast<Mangled::ManglingScheme>(
      Mangled::eFirstRegisteredManglingScheme + index);
}

/// Returns true if a name could start with both one of \p prefixes and a
/// prefix of one of the active \p schemes.
[[maybe_unused]] static bool
OverlapsActiveScheme(llvm::ArrayRef<llvm::StringRef> prefixes,
                     llvm::ArrayRef<RegisteredManglingScheme> schemes) {
  for (const RegisteredManglingScheme &scheme : schemes) {
    if (!scheme.active.load(std::memory_order_relaxed))
      continue;
    for (llvm::StringRef prefix : prefixes)
      for (llvm::StringRef other : scheme.prefixes)
        if (prefix.starts_with(other) || other.starts_with(prefix))
          return true;
  }
  return false;
}

Mangled::ManglingScheme
ManglingSchemeRegistry::Register(llvm::ArrayRef<llvm::StringRef> prefixes,
                                 lldb::LanguageType language,
                                 Mangled::ManglingSchemeMatcher matches) {
  std::lock_guard<std::mutex> guard(m_mutex);
  const size_t num_schemes = m_num_schemes.load(std::memory_order_relaxed);
  llvm::ArrayRef<RegisteredManglingScheme> schemes(m_schemes.data(),
                                                   num_schemes);
  auto it = llvm::find_if(schemes, [&](const RegisteredManglingScheme &scheme) {
    return scheme.language == language && scheme.matches == matches &&
           llvm::equal(scheme.prefixes, prefixes);
  });
  const size_t index = std::distance(schemes.begin(), it);

  if (it == schemes.end()) {
    assert(index < m_schemes.size() && "too many mangling schemes");
    if (index >= m_schemes.size())
      return Mangled::eManglingSchemeNone;
    assert(!OverlapsActiveScheme(prefixes, schemes) &&
           "mangling scheme prefixes overlap");
    assert(
        prefixes.size() <= RegisteredManglingScheme::g_inline_prefixes &&
        llvm::all_of(prefixes,
                     [](llvm::StringRef prefix) {
                       return prefix.size() <=
                              RegisteredManglingScheme::g_inline_prefix_length;
                     }) &&
        "mangling scheme prefixes don't fit inline");
    m_schemes[index].prefixes.assign(prefixes.begin(), prefixes.end());
    m_schemes[index].matches = matches;
    m_schemes[index].language = language;
    m_num_schemes.store(num_schemes + 1, std::memory_order_release);
  }
  m_schemes[index].active.store(true, std::memory_order_relaxed);
  return SchemeForIndex(index);
}

void ManglingSchemeRegistry::Unregister(Mangled::ManglingScheme scheme) {
  if (scheme < Mangled::eFirstRegisteredManglingScheme)
    return;
  std::lock_guard<std::mutex> guard(m_mutex);
  size_t index = scheme - Mangled::eFirstRegisteredManglingScheme;
  if (index < m_num_schemes.load(std::memory_order_relaxed))
    m_schemes[index].active.store(false, std::memory_order_relaxed);
}

lldb::LanguageType
ManglingSchemeRegistry::GetLanguage(Mangled::ManglingScheme scheme) const {
  if (scheme < Mangled::eFirstRegisteredManglingScheme)
    return lldb::eLanguageTypeUnknown;
  size_t index = scheme - Mangled::eFirstRegisteredManglingScheme;
  if (index >= m_num_schemes.load(std::memory_order_acquire))
    return lldb::eLanguageTypeUnknown;
  return m_schemes[index].language;
}

Mangled::ManglingScheme
ManglingSchemeRegistry::Find(llvm::StringRef name) const {
  const size_t num_schemes = m_num_schemes.load(std::memory_order_acquire);
  for (auto [index, scheme] :
       llvm::enumerate(llvm::ArrayRef(m_schemes.data(), num_schemes))) {
    if (!scheme.active.load(std::memory_order_relaxed) ||
        !llvm::any_of(scheme.prefixes, [&](llvm::StringRef prefix) {
          return name.starts_with(prefix);
        }))
      continue;
    // Prefixes don't overlap, so no other scheme can match the name.
    if (scheme.matches && !scheme.matches(name))
      return Mangled::eManglingSchemeNone;
    return SchemeForIndex(index);
  }
  return Mangled::eManglingSchemeNone;
}

Mangled::ManglingScheme
Mangled::RegisterManglingScheme(llvm::ArrayRef<llvm::StringRef> prefixes,
                                lldb::LanguageType language,
                                ManglingSchemeMatcher matches) {
  return GetManglingSchemeRegistry().Register(prefixes, language, matches);
}

void Mangled::UnregisterManglingScheme(ManglingScheme scheme) {
  GetManglingSchemeRegistry().Unregister(scheme);
}

lldb::LanguageType
Mangled::GetLanguageForManglingScheme(ManglingScheme scheme) {
  return GetManglingSchemeRegistry().GetLanguage(scheme);
}

bool Mangled::IsMangledName(llvm::StringRef name) {
  return Mangled::GetManglingScheme(name) != Mangled::eManglingSchemeNone;
}

Mangled::ManglingScheme Mangled::GetManglingScheme(llvm::StringRef name) {
  if (name.empty())
    return Mangled::eManglingSchemeNone;

  // Swift's older style of mangling used "_T" as a mangling prefix. This can
  // lead to false positives with other symbols that just so happen to start
  // with "_T". To minimize the chance of that happening, we only return true
  // for select old-style swift mangled names. The known cases are ObjC classes
  // and protocols. Classes are either prefixed with "_TtC" or "_TtGC".
  // Protocols are prefixed with "_TtP".
  if (name.starts_with("_TtC") || name.starts_with("_TtGC") ||
      name.starts_with("_TtP"))
    return Mangled::eManglingSchemeSwift;

  // Swift 4.2 used "$S" and "_$S".
  // Swift 5 and onward uses "$s" and "_$s".
  // Swift also uses "@__swiftmacro_" as a prefix for mangling filenames.
  // Embedded Swift introduced "$e" and  "_$e" as Swift mangling prefixes.
  if (name.starts_with("$S") || name.starts_with("_$S") ||
      name.starts_with("$s") || name.starts_with("_$s") ||
      name.starts_with("$e") || name.starts_with("_$e") ||
      name.starts_with("@__swiftmacro_"))
    return Mangled::eManglingSchemeSwift;

  return GetManglingSchemeRegistry().Find(name);
}

Mangled::Mangled(ConstString s) : m_mangled(), m_demangled() {
  if (s)
    SetValue(s);
}

Mangled::Mangled(llvm::StringRef name) {
  if (!name.empty())
    SetValue(ConstString(name));
}

// Convert to bool operator. This allows code to check any Mangled objects
// to see if they contain anything valid using code such as:
//
//  Mangled mangled(...);
//  if (mangled)
//  { ...
Mangled::operator bool() const { return m_mangled || m_demangled; }

// Clear the mangled and demangled values.
void Mangled::Clear() {
  m_mangled.Clear();
  m_demangled.Clear();
}

// Compare the string values.
int Mangled::Compare(const Mangled &a, const Mangled &b) {
  return ConstString::Compare(a.GetName(ePreferMangled),
                              b.GetName(ePreferMangled));
}

void Mangled::SetValue(ConstString name) {
  if (name) {
    if (IsMangledName(name.GetStringRef())) {
      m_demangled.Clear();
      m_mangled = name;
    } else {
      m_demangled = name;
      m_mangled.Clear();
    }
  } else {
    m_demangled.Clear();
    m_mangled.Clear();
  }
}

// Generate the demangled name on demand using this accessor. Code in this
// class will need to use this accessor if it wishes to decode the demangled
// name. The result is cached and will be kept until a new string value is
// supplied to this object, or until the end of the object's lifetime.
ConstString Mangled::GetDemangledName() const {
  if (!m_mangled)
    return m_demangled;

  // Re-use previously demangled names.
  if (!m_demangled.IsNull())
    return m_demangled;

  if (m_mangled.GetMangledCounterpart(m_demangled) && !m_demangled.IsNull())
    return m_demangled;

  // We didn't already mangle this name, demangle it and if all goes well
  // add it to our map.
  ManglingScheme scheme = GetManglingScheme(m_mangled.GetStringRef());
  switch (scheme) {
  case eManglingSchemeSwift:
    // Demangling a swift name requires the swift compiler. This is
    // explicitly unsupported on llvm.org.
    break;
  case eManglingSchemeNone:
    // Don't bother demangling anything that isn't mangled.
    break;
  default:
    // Language plugins demangle the schemes they registered.
    if (Language *lang = Language::FindPluginForManglingScheme(scheme)) {
      Language::DemangledName demangled =
          lang->DemangleSymbol(m_mangled, scheme, Language::DemangleOptions());
      if (!demangled.name.empty())
        m_demangled.SetStringWithMangledCounterpart(demangled.name, m_mangled);
    }
    break;
  }

  if (m_demangled.IsNull()) {
    // Set the demangled string to the empty string to indicate we tried to
    // parse it once and failed.
    m_demangled.SetCString("");
  }

  return m_demangled;
}

std::optional<DemangledNameInfo> Mangled::ComputeDemangledInfo() const {
  if (!m_mangled)
    return std::nullopt;

  // Only the language plugins compute the info, for the schemes they
  // registered.
  ManglingScheme scheme = GetManglingScheme(m_mangled.GetStringRef());
  Language *lang = Language::FindPluginForManglingScheme(scheme);
  if (!lang)
    return std::nullopt;

  Language::DemangleOptions options;
  options.want_info = true;
  return lang->DemangleSymbol(m_mangled, scheme, options).info;
}

ConstString Mangled::GetDisplayDemangledName() const {
  if (Language *lang = Language::FindPlugin(GuessLanguage()))
    return lang->GetDisplayDemangledName(*this);
  return GetDemangledName();
}

bool Mangled::NameMatches(const RegularExpression &regex) const {
  if (m_mangled && regex.Execute(m_mangled.GetStringRef()))
    return true;

  ConstString demangled = GetDemangledName();
  return demangled && regex.Execute(demangled.GetStringRef());
}

// Get the demangled name if there is one, else return the mangled name.
ConstString Mangled::GetName(Mangled::NamePreference preference) const {
  if (preference == ePreferMangled && m_mangled)
    return m_mangled;

  // Call the accessor to make sure we get a demangled name in case it hasn't
  // been demangled yet...
  ConstString demangled = GetDemangledName();

  if (preference == ePreferDemangledWithoutArguments) {
    if (Language *lang = Language::FindPlugin(GuessLanguage())) {
      return lang->GetDemangledFunctionNameWithoutArguments(*this);
    }
  }
  if (preference == ePreferDemangled) {
    if (demangled)
      return demangled;
    return m_mangled;
  }
  return demangled;
}

// Dump a Mangled object to stream "s". We don't force our demangled name to be
// computed currently (we don't use the accessor).
void Mangled::Dump(Stream *s) const {
  if (m_mangled) {
    *s << ", mangled = " << m_mangled;
  }
  if (m_demangled)
    s->Format(", demangled = {0}", m_demangled.GetStringRef());
}

// Dumps a debug version of this string with extra object and state information
// to stream "s".
void Mangled::DumpDebug(Stream *s) const {
  s->Printf("%*p: Mangled mangled = ", static_cast<int>(sizeof(void *) * 2),
            static_cast<const void *>(this));
  m_mangled.DumpDebug(s);
  s->PutCString(", demangled = ");
  m_demangled.DumpDebug(s);
}

// We "guess" the language because we can't determine a symbol's language from
// it's name.  For example, a Pascal symbol can be mangled using the C++
// Itanium scheme, and defined in a compilation unit within the same module as
// other C++ units.  In addition, different targets could have different ways
// of mangling names from a given language, likewise the compilation units
// within those targets.
lldb::LanguageType Mangled::GuessLanguage() const {
  // The language that registered the mangling scheme knows the name best.
  if (m_mangled) {
    lldb::LanguageType language = GetLanguageForManglingScheme(
        GetManglingScheme(m_mangled.GetStringRef()));
    if (language != lldb::eLanguageTypeUnknown)
      return language;
  }

  lldb::LanguageType result = lldb::eLanguageTypeUnknown;
  // Ask each language plugin to check if the mangled name belongs to it.
  Language::ForEach([this, &result](Language *l) {
    if (l->SymbolNameFitsToLanguage(*this)) {
      result = l->GetLanguageType();
      return IterationAction::Stop;
    }
    return IterationAction::Continue;
  });
  return result;
}

// Dump OBJ to the supplied stream S.
Stream &operator<<(Stream &s, const Mangled &obj) {
  if (obj.GetMangledName())
    s << "mangled = '" << obj.GetMangledName() << "'";

  ConstString demangled = obj.GetDemangledName();
  if (demangled)
    s << ", demangled = '" << demangled << '\'';
  else
    s << ", demangled = <error>";
  return s;
}

// When encoding Mangled objects we can get away with encoding as little
// information as is required. The enumeration below helps us to efficiently
// encode Mangled objects.
enum MangledEncoding {
  /// If the Mangled object has neither a mangled name or demangled name we can
  /// encode the object with one zero byte using the Empty enumeration.
  Empty = 0u,
  /// If the Mangled object has only a demangled name and no mangled named, we
  /// can encode only the demangled name.
  DemangledOnly = 1u,
  /// If the mangle name can calculate the demangled name (it is the
  /// mangled/demangled counterpart), then we only need to encode the mangled
  /// name as the demangled name can be recomputed.
  MangledOnly = 2u,
  /// If we have a Mangled object with two different names that are not related
  /// then we need to save both strings. This can happen if we have a name that
  /// isn't a true mangled name, but we want to be able to lookup a symbol by
  /// name and type in the symbol table. We do this for Objective C symbols like
  /// "OBJC_CLASS_$_NSValue" where the mangled named will be set to
  /// "OBJC_CLASS_$_NSValue" and the demangled name will be manually set to
  /// "NSValue". If we tried to demangled the name "OBJC_CLASS_$_NSValue" it
  /// would fail, but in these cases we want these unrelated names to be
  /// preserved.
  MangledAndDemangled = 3u
};

bool Mangled::Decode(const DataExtractor &data, lldb::offset_t *offset_ptr,
                     const StringTableReader &strtab) {
  m_mangled.Clear();
  m_demangled.Clear();
  MangledEncoding encoding = (MangledEncoding)data.GetU8(offset_ptr);
  switch (encoding) {
    case Empty:
      return true;

    case DemangledOnly:
      m_demangled.SetString(strtab.Get(data.GetU32(offset_ptr)));
      return true;

    case MangledOnly:
      m_mangled.SetString(strtab.Get(data.GetU32(offset_ptr)));
      return true;

    case MangledAndDemangled:
      m_mangled.SetString(strtab.Get(data.GetU32(offset_ptr)));
      m_demangled.SetString(strtab.Get(data.GetU32(offset_ptr)));
      return true;
  }
  return false;
}
/// The encoding format for the Mangled object is as follows:
///
/// uint8_t encoding;
/// char str1[]; (only if DemangledOnly, MangledOnly)
/// char str2[]; (only if MangledAndDemangled)
///
/// The strings are stored as null-terminated UTF8 strings and str1 and str2
/// are only saved if we need them based on the encoding.
///
/// Some mangled names have a mangled name that can be demangled by the built
/// in demanglers. These kinds of mangled objects know when the mangled and
/// demangled names are the counterparts for each other. This is done because
/// demangling is very expensive and avoiding demangling the same name twice
/// saves us a lot of compute time. For these kinds of names we only need to
/// save the mangled name and have the encoding set to "MangledOnly".
///
/// If a mangled object has only a demangled name, then we save only that string
/// and have the encoding set to "DemangledOnly".
///
/// Some mangled objects have both mangled and demangled names, but the
/// demangled name can not be computed from the mangled name. This is often used
/// for runtime named, like Objective C runtime V2 and V3 names. Both these
/// names must be saved and the encoding is set to "MangledAndDemangled".
///
/// For a Mangled object with no names, we only need to set the encoding to
/// "Empty" and not store any string values.
void Mangled::Encode(DataEncoder &file, ConstStringTable &strtab) const {
  MangledEncoding encoding = Empty;
  if (m_mangled) {
    encoding = MangledOnly;
    if (m_demangled) {
      // We have both mangled and demangled names. If the demangled name is the
      // counterpart of the mangled name, then we only need to save the mangled
      // name. If they are different, we need to save both.
      ConstString s;
      if (!(m_mangled.GetMangledCounterpart(s) && s == m_demangled))
        encoding = MangledAndDemangled;
    }
  } else if (m_demangled) {
    encoding = DemangledOnly;
  }
  file.AppendU8(encoding);
  switch (encoding) {
    case Empty:
      break;
    case DemangledOnly:
      file.AppendU32(strtab.Add(m_demangled));
      break;
    case MangledOnly:
      file.AppendU32(strtab.Add(m_mangled));
      break;
    case MangledAndDemangled:
      file.AppendU32(strtab.Add(m_mangled));
      file.AppendU32(strtab.Add(m_demangled));
      break;
  }
}

ConstString Mangled::GetBaseName() const {
  std::optional<DemangledNameInfo> demangled_info = ComputeDemangledInfo();
  if (!demangled_info)
    return {};

  ConstString demangled_name = GetDemangledName();
  if (!demangled_name)
    return {};

  const auto &range = demangled_info->BasenameRange;
  return ConstString(
      demangled_name.GetStringRef().slice(range.first, range.second));
}
