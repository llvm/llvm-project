//===- Multilib.cpp - Multilib Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Multilib.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <string>

using namespace clang;
using namespace driver;
using namespace llvm::sys;

Multilib::Multilib(StringRef GCCSuffix, StringRef OSSuffix,
                   StringRef IncludeSuffix, const flags_list &Flags)
    : GCCSuffix(GCCSuffix), OSSuffix(OSSuffix), IncludeSuffix(IncludeSuffix),
      Flags(Flags) {
  assert(GCCSuffix.empty() ||
         (StringRef(GCCSuffix).front() == '/' && GCCSuffix.size() > 1));
  assert(OSSuffix.empty() ||
         (StringRef(OSSuffix).front() == '/' && OSSuffix.size() > 1));
  assert(IncludeSuffix.empty() ||
         (StringRef(IncludeSuffix).front() == '/' && IncludeSuffix.size() > 1));
}

LLVM_DUMP_METHOD void Multilib::dump() const {
  print(llvm::errs());
}

void Multilib::print(raw_ostream &OS) const {
  if (GCCSuffix.empty())
    OS << ".";
  else {
    OS << StringRef(GCCSuffix).drop_front();
  }
  OS << ";";
  for (StringRef Flag : Flags) {
    if (Flag.front() == '-')
      OS << "@" << Flag.substr(1);
  }
}

bool Multilib::operator==(const Multilib &Other) const {
  // Check whether the flags sets match
  // allowing for the match to be order invariant
  llvm::StringSet<> MyFlags;
  for (const auto &Flag : Flags)
    MyFlags.insert(Flag);

  for (const auto &Flag : Other.Flags)
    if (!MyFlags.contains(Flag))
      return false;

  if (osSuffix() != Other.osSuffix())
    return false;

  if (gccSuffix() != Other.gccSuffix())
    return false;

  if (includeSuffix() != Other.includeSuffix())
    return false;

  return true;
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const Multilib &M) {
  M.print(OS);
  return OS;
}

MultilibSet &MultilibSet::FilterOut(FilterCallback F) {
  llvm::erase_if(Multilibs, F);
  return *this;
}

void MultilibSet::push_back(const Multilib &M) { Multilibs.push_back(M); }

bool MultilibSet::select(const Multilib::flags_list &Flags,
                         llvm::SmallVector<Multilib> &Selected) const {
  llvm::StringSet<> FlagSet(expandFlags(Flags));
  Selected.clear();
  llvm::copy_if(Multilibs, std::back_inserter(Selected),
                [&FlagSet](const Multilib &M) {
                  for (const std::string &F : M.flags())
                    if (!FlagSet.contains(F))
                      return false;
                  return true;
                });
  return !Selected.empty();
}

llvm::StringSet<>
MultilibSet::expandFlags(const Multilib::flags_list &InFlags) const {
  llvm::StringSet<> Result;
  for (const auto &F : InFlags)
    Result.insert(F);
  for (const FlagMatcher &M : FlagMatchers) {
    std::string RegexString(M.Match);

    // Make the regular expression match the whole string.
    if (!StringRef(M.Match).starts_with("^"))
      RegexString.insert(RegexString.begin(), '^');
    if (!StringRef(M.Match).ends_with("$"))
      RegexString.push_back('$');

    const llvm::Regex Regex(RegexString);
    assert(Regex.isValid());
    if (llvm::find_if(InFlags, [&Regex](StringRef F) {
          return Regex.match(F);
        }) != InFlags.end()) {
      Result.insert(M.Flags.begin(), M.Flags.end());
    }
  }
  return Result;
}

namespace {

// When updating this also update MULTILIB_VERSION in MultilibTest.cpp
static const VersionTuple MultilibVersionCurrent(1, 0);

struct MultilibSerialization {
  std::string Dir;
  std::vector<std::string> Flags;
};

struct MultilibSetSerialization {
  llvm::VersionTuple MultilibVersion;
  std::vector<MultilibSerialization> Multilibs;
  std::vector<MultilibSet::FlagMatcher> FlagMatchers;
};

} // end anonymous namespace

template <> struct llvm::yaml::MappingTraits<MultilibSerialization> {
  static void mapping(llvm::yaml::IO &io, MultilibSerialization &V) {
    io.mapRequired("Dir", V.Dir);
    io.mapRequired("Flags", V.Flags);
  }
  static std::string validate(IO &io, MultilibSerialization &V) {
    if (StringRef(V.Dir).starts_with("/"))
      return "paths must be relative but \"" + V.Dir + "\" starts with \"/\"";
    return std::string{};
  }
};

template <> struct llvm::yaml::MappingTraits<MultilibSet::FlagMatcher> {
  static void mapping(llvm::yaml::IO &io, MultilibSet::FlagMatcher &M) {
    io.mapRequired("Match", M.Match);
    io.mapRequired("Flags", M.Flags);
  }
  static std::string validate(IO &io, MultilibSet::FlagMatcher &M) {
    llvm::Regex Regex(M.Match);
    std::string RegexError;
    if (!Regex.isValid(RegexError))
      return RegexError;
    if (M.Flags.empty())
      return "value required for 'Flags'";
    return std::string{};
  }
};

template <> struct llvm::yaml::MappingTraits<MultilibSetSerialization> {
  static void mapping(llvm::yaml::IO &io, MultilibSetSerialization &M) {
    io.mapRequired("MultilibVersion", M.MultilibVersion);
    io.mapRequired("Variants", M.Multilibs);
    io.mapOptional("Mappings", M.FlagMatchers);
  }
  static std::string validate(IO &io, MultilibSetSerialization &M) {
    if (M.MultilibVersion.empty())
      return "missing required key 'MultilibVersion'";
    if (M.MultilibVersion.getMajor() != MultilibVersionCurrent.getMajor())
      return "multilib version " + M.MultilibVersion.getAsString() +
             " is unsupported";
    if (M.MultilibVersion.getMinor() > MultilibVersionCurrent.getMinor())
      return "multilib version " + M.MultilibVersion.getAsString() +
             " is unsupported";
    return std::string{};
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(MultilibSerialization)
LLVM_YAML_IS_SEQUENCE_VECTOR(MultilibSet::FlagMatcher)

llvm::ErrorOr<MultilibSet>
MultilibSet::parseYaml(llvm::MemoryBufferRef Input,
                       llvm::SourceMgr::DiagHandlerTy DiagHandler,
                       void *DiagHandlerCtxt) {
  MultilibSetSerialization MS;
  llvm::yaml::Input YamlInput(Input, nullptr, DiagHandler, DiagHandlerCtxt);
  YamlInput >> MS;
  if (YamlInput.error())
    return YamlInput.error();

  multilib_list Multilibs;
  Multilibs.reserve(MS.Multilibs.size());
  for (const auto &M : MS.Multilibs) {
    std::string Dir;
    if (M.Dir != ".")
      Dir = "/" + M.Dir;
    Multilibs.emplace_back(Dir, Dir, Dir, M.Flags);
  }

  return MultilibSet(std::move(Multilibs), std::move(MS.FlagMatchers));
}

LLVM_DUMP_METHOD void MultilibSet::dump() const {
  print(llvm::errs());
}

void MultilibSet::print(raw_ostream &OS) const {
  for (const auto &M : *this)
    OS << M << "\n";
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const MultilibSet &MS) {
  MS.print(OS);
  return OS;
}
