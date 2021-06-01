//===-- Mangled.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Mangled.h"

#include "lldb/Core/RichManglingContext.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-enumerations.h"

#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Compiler.h"

#include <mutex>
#include <string>
#include <utility>

#include <cstdlib>
#include <cstring>
using namespace lldb_private;

static inline bool cstring_is_mangled(llvm::StringRef s) {
  return Mangled::GetManglingScheme(s) != Mangled::eManglingSchemeNone;
}

static ConstString GetDemangledNameWithoutArguments(ConstString mangled,
                                                    ConstString demangled) {
  const char *mangled_name_cstr = mangled.GetCString();

  if (demangled && mangled_name_cstr && mangled_name_cstr[0]) {
    if (mangled_name_cstr[0] == '_' && mangled_name_cstr[1] == 'Z' &&
        (mangled_name_cstr[2] != 'T' && // avoid virtual table, VTT structure,
                                        // typeinfo structure, and typeinfo
                                        // mangled_name
         mangled_name_cstr[2] != 'G' && // avoid guard variables
         mangled_name_cstr[2] != 'Z')) // named local entities (if we eventually
                                       // handle eSymbolTypeData, we will want
                                       // this back)
    {
      CPlusPlusLanguage::MethodName cxx_method(demangled);
      if (!cxx_method.GetBasename().empty()) {
        std::string shortname;
        if (!cxx_method.GetContext().empty())
          shortname = cxx_method.GetContext().str() + "::";
        shortname += cxx_method.GetBasename().str();
        return ConstString(shortname);
      }
    }
  }
  if (demangled)
    return demangled;
  return mangled;
}

#pragma mark Mangled

Mangled::ManglingScheme Mangled::GetManglingScheme(llvm::StringRef const name) {
  if (name.empty())
    return Mangled::eManglingSchemeNone;

  if (name.startswith("?"))
    return Mangled::eManglingSchemeMSVC;

  if (name.startswith("_Z"))
    return Mangled::eManglingSchemeItanium;

  // ___Z is a clang extension of block invocations
  if (name.startswith("___Z"))
    return Mangled::eManglingSchemeItanium;

  return Mangled::eManglingSchemeNone;
}

Mangled::Mangled(ConstString s) : m_mangled(), m_demangled() {
  if (s)
    SetValue(s);
}

Mangled::Mangled(llvm::StringRef name) {
  if (!name.empty())
    SetValue(ConstString(name));
}

// Convert to pointer operator. This allows code to check any Mangled objects
// to see if they contain anything valid using code such as:
//
//  Mangled mangled(...);
//  if (mangled)
//  { ...
Mangled::operator void *() const {
  return (m_mangled) ? const_cast<Mangled *>(this) : nullptr;
}

// Logical NOT operator. This allows code to check any Mangled objects to see
// if they are invalid using code such as:
//
//  Mangled mangled(...);
//  if (!file_spec)
//  { ...
bool Mangled::operator!() const { return !m_mangled; }

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

// Set the string value in this objects. If "mangled" is true, then the mangled
// named is set with the new value in "s", else the demangled name is set.
void Mangled::SetValue(ConstString s, bool mangled) {
  if (s) {
    if (mangled) {
      m_demangled.Clear();
      m_mangled = s;
    } else {
      m_demangled = s;
      m_mangled.Clear();
    }
  } else {
    m_demangled.Clear();
    m_mangled.Clear();
  }
}

void Mangled::SetValue(ConstString name) {
  if (name) {
    if (cstring_is_mangled(name.GetStringRef())) {
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

// Local helpers for different demangling implementations.
static char *GetMSVCDemangledStr(const char *M) {
  char *demangled_cstr = llvm::microsoftDemangle(
      M, nullptr, nullptr, nullptr, nullptr,
      llvm::MSDemangleFlags(llvm::MSDF_NoAccessSpecifier |
                            llvm::MSDF_NoCallingConvention |
                            llvm::MSDF_NoMemberType));

  if (Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_DEMANGLE)) {
    if (demangled_cstr && demangled_cstr[0])
      LLDB_LOGF(log, "demangled msvc: %s -> \"%s\"", M, demangled_cstr);
    else
      LLDB_LOGF(log, "demangled msvc: %s -> error", M);
  }

  return demangled_cstr;
}

static char *GetItaniumDemangledStr(const char *M) {
  char *demangled_cstr = nullptr;

  llvm::ItaniumPartialDemangler ipd;
  bool err = ipd.partialDemangle(M);
  if (!err) {
    // Default buffer and size (will realloc in case it's too small).
    size_t demangled_size = 80;
    demangled_cstr = static_cast<char *>(std::malloc(demangled_size));
    demangled_cstr = ipd.finishDemangle(demangled_cstr, &demangled_size);

    assert(demangled_cstr &&
           "finishDemangle must always succeed if partialDemangle did");
    assert(demangled_cstr[demangled_size - 1] == '\0' &&
           "Expected demangled_size to return length including trailing null");
  }

  if (Log *log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_DEMANGLE)) {
    if (demangled_cstr)
      LLDB_LOGF(log, "demangled itanium: %s -> \"%s\"", M, demangled_cstr);
    else
      LLDB_LOGF(log, "demangled itanium: %s -> error: failed to demangle", M);
  }

  return demangled_cstr;
}

// Explicit demangling for scheduled requests during batch processing. This
// makes use of ItaniumPartialDemangler's rich demangle info
bool Mangled::DemangleWithRichManglingInfo(
    RichManglingContext &context, SkipMangledNameFn *skip_mangled_name) {
  // Others are not meant to arrive here. ObjC names or C's main() for example
  // have their names stored in m_demangled, while m_mangled is empty.
  assert(m_mangled);

  // Check whether or not we are interested in this name at all.
  ManglingScheme scheme = GetManglingScheme(m_mangled.GetStringRef());
  if (skip_mangled_name && skip_mangled_name(m_mangled.GetStringRef(), scheme))
    return false;

  switch (scheme) {
  case eManglingSchemeNone:
    // The current mangled_name_filter would allow llvm_unreachable here.
    return false;

  case eManglingSchemeItanium:
    // We want the rich mangling info here, so we don't care whether or not
    // there is a demangled string in the pool already.
    if (context.FromItaniumName(m_mangled)) {
      // If we got an info, we have a name. Copy to string pool and connect the
      // counterparts to accelerate later access in GetDemangledName().
      context.ParseFullName();
      m_demangled.SetStringWithMangledCounterpart(context.GetBufferRef(),
                                                  m_mangled);
      return true;
    } else {
      m_demangled.SetCString("");
      return false;
    }

  case eManglingSchemeMSVC: {
    // We have no rich mangling for MSVC-mangled names yet, so first try to
    // demangle it if necessary.
    if (!m_demangled && !m_mangled.GetMangledCounterpart(m_demangled)) {
      if (char *d = GetMSVCDemangledStr(m_mangled.GetCString())) {
        // If we got an info, we have a name. Copy to string pool and connect
        // the counterparts to accelerate later access in GetDemangledName().
        m_demangled.SetStringWithMangledCounterpart(llvm::StringRef(d),
                                                    m_mangled);
        ::free(d);
      } else {
        m_demangled.SetCString("");
      }
    }

    if (m_demangled.IsEmpty()) {
      // Cannot demangle it, so don't try parsing.
      return false;
    } else {
      // Demangled successfully, we can try and parse it with
      // CPlusPlusLanguage::MethodName.
      return context.FromCxxMethodName(m_demangled);
    }
  }
  }
  llvm_unreachable("Fully covered switch above!");
}

// Generate the demangled name on demand using this accessor. Code in this
// class will need to use this accessor if it wishes to decode the demangled
// name. The result is cached and will be kept until a new string value is
// supplied to this object, or until the end of the object's lifetime.
ConstString Mangled::GetDemangledName() const {
  // Check to make sure we have a valid mangled name and that we haven't
  // already decoded our mangled name.
  if (m_mangled && m_demangled.IsNull()) {
    // Don't bother running anything that isn't mangled
    const char *mangled_name = m_mangled.GetCString();
    ManglingScheme mangling_scheme = GetManglingScheme(m_mangled.GetStringRef());
    if (mangling_scheme != eManglingSchemeNone &&
        !m_mangled.GetMangledCounterpart(m_demangled)) {
      // We didn't already mangle this name, demangle it and if all goes well
      // add it to our map.
      char *demangled_name = nullptr;
      switch (mangling_scheme) {
      case eManglingSchemeMSVC:
        demangled_name = GetMSVCDemangledStr(mangled_name);
        break;
      case eManglingSchemeItanium: {
        demangled_name = GetItaniumDemangledStr(mangled_name);
        break;
      }
      case eManglingSchemeNone:
        llvm_unreachable("eManglingSchemeNone was handled already");
      }
      if (demangled_name) {
        m_demangled.SetStringWithMangledCounterpart(
            llvm::StringRef(demangled_name), m_mangled);
        free(demangled_name);
      }
    }
    if (m_demangled.IsNull()) {
      // Set the demangled string to the empty string to indicate we tried to
      // parse it once and failed.
      m_demangled.SetCString("");
    }
  }

  return m_demangled;
}

ConstString
Mangled::GetDisplayDemangledName() const {
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

  ConstString demangled = GetDemangledName();

  if (preference == ePreferDemangledWithoutArguments) {
    return GetDemangledNameWithoutArguments(m_mangled, demangled);
  }
  if (preference == ePreferDemangled) {
    // Call the accessor to make sure we get a demangled name in case it hasn't
    // been demangled yet...
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
  if (m_demangled) {
    const char *demangled = m_demangled.AsCString();
    s->Printf(", demangled = %s", demangled[0] ? demangled : "<error>");
  }
}

// Dumps a debug version of this string with extra object and state information
// to stream "s".
void Mangled::DumpDebug(Stream *s) const {
  s->Printf("%*p: Mangled mangled = ", static_cast<int>(sizeof(void *) * 2),
            static_cast<const void *>(this));
  m_mangled.DumpDebug(s);
  s->Printf(", demangled = ");
  m_demangled.DumpDebug(s);
}

// Return the size in byte that this object takes in memory. The size includes
// the size of the objects it owns, and not the strings that it references
// because they are shared strings.
size_t Mangled::MemorySize() const {
  return m_mangled.MemorySize() + m_demangled.MemorySize();
}

// We "guess" the language because we can't determine a symbol's language from
// it's name.  For example, a Pascal symbol can be mangled using the C++
// Itanium scheme, and defined in a compilation unit within the same module as
// other C++ units.  In addition, different targets could have different ways
// of mangling names from a given language, likewise the compilation units
// within those targets.
lldb::LanguageType Mangled::GuessLanguage() const {
  lldb::LanguageType result = lldb::eLanguageTypeUnknown;
  // Ask each language plugin to check if the mangled name belongs to it.
  Language::ForEach([this, &result](Language *l) {
    if (l->SymbolNameFitsToLanguage(*this)) {
      result = l->GetLanguageType();
      return false;
    }
    return true;
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
