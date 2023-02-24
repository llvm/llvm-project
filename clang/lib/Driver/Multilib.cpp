//===- Multilib.cpp - Multilib Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Multilib.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <string>

using namespace clang;
using namespace driver;
using namespace llvm::sys;

Multilib::Multilib(StringRef GCCSuffix, StringRef OSSuffix,
                   StringRef IncludeSuffix, int Priority,
                   const flags_list &Flags)
    : GCCSuffix(GCCSuffix), OSSuffix(OSSuffix), IncludeSuffix(IncludeSuffix),
      Flags(Flags), Priority(Priority) {
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
    if (Flag.front() == '+')
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
    if (MyFlags.find(Flag) == MyFlags.end())
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
  filterInPlace(F, Multilibs);
  return *this;
}

void MultilibSet::push_back(const Multilib &M) { Multilibs.push_back(M); }

static bool isFlagEnabled(StringRef Flag) {
  char Indicator = Flag.front();
  assert(Indicator == '+' || Indicator == '-');
  return Indicator == '+';
}

bool MultilibSet::select(const Multilib::flags_list &Flags, Multilib &M) const {
  llvm::StringMap<bool> FlagSet;

  // Stuff all of the flags into the FlagSet such that a true mappend indicates
  // the flag was enabled, and a false mappend indicates the flag was disabled.
  for (StringRef Flag : Flags)
    FlagSet[Flag.substr(1)] = isFlagEnabled(Flag);

  multilib_list Filtered = filterCopy([&FlagSet](const Multilib &M) {
    for (StringRef Flag : M.flags()) {
      llvm::StringMap<bool>::const_iterator SI = FlagSet.find(Flag.substr(1));
      if (SI != FlagSet.end())
        if (SI->getValue() != isFlagEnabled(Flag))
          return true;
    }
    return false;
  }, Multilibs);

  if (Filtered.empty())
    return false;
  if (Filtered.size() == 1) {
    M = Filtered[0];
    return true;
  }

  // Sort multilibs by priority and select the one with the highest priority.
  llvm::sort(Filtered, [](const Multilib &a, const Multilib &b) -> bool {
    return a.priority() > b.priority();
  });

  if (Filtered[0].priority() > Filtered[1].priority()) {
    M = Filtered[0];
    return true;
  }

  // TODO: We should consider returning llvm::Error rather than aborting.
  assert(false && "More than one multilib with the same priority");
  return false;
}

LLVM_DUMP_METHOD void MultilibSet::dump() const {
  print(llvm::errs());
}

void MultilibSet::print(raw_ostream &OS) const {
  for (const auto &M : *this)
    OS << M << "\n";
}

MultilibSet::multilib_list MultilibSet::filterCopy(FilterCallback F,
                                                   const multilib_list &Ms) {
  multilib_list Copy(Ms);
  filterInPlace(F, Copy);
  return Copy;
}

void MultilibSet::filterInPlace(FilterCallback F, multilib_list &Ms) {
  llvm::erase_if(Ms, F);
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const MultilibSet &MS) {
  MS.print(OS);
  return OS;
}
