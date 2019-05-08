//===- AnalyzerOptions.cpp - Analysis Engine Options ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains special accessors for analyzer configuration options
// with string representations.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

using namespace clang;
using namespace ento;
using namespace llvm;

std::vector<StringRef>
AnalyzerOptions::getRegisteredCheckers(bool IncludeExperimental /* = false */) {
  static const StringRef StaticAnalyzerChecks[] = {
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, HELPTEXT, DOC_URI, IS_HIDDEN)                 \
  FULLNAME,
#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS
  };
  std::vector<StringRef> Result;
  for (StringRef CheckName : StaticAnalyzerChecks) {
    if (!CheckName.startswith("debug.") &&
        (IncludeExperimental || !CheckName.startswith("alpha.")))
      Result.push_back(CheckName);
  }
  return Result;
}

ExplorationStrategyKind
AnalyzerOptions::getExplorationStrategy() const {
  auto K =
    llvm::StringSwitch<llvm::Optional<ExplorationStrategyKind>>(
                                                            ExplorationStrategy)
          .Case("dfs", ExplorationStrategyKind::DFS)
          .Case("bfs", ExplorationStrategyKind::BFS)
          .Case("unexplored_first",
                ExplorationStrategyKind::UnexploredFirst)
          .Case("unexplored_first_queue",
                ExplorationStrategyKind::UnexploredFirstQueue)
          .Case("unexplored_first_location_queue",
                ExplorationStrategyKind::UnexploredFirstLocationQueue)
          .Case("bfs_block_dfs_contents",
                ExplorationStrategyKind::BFSBlockDFSContents)
          .Default(None);
  assert(K.hasValue() && "User mode is invalid.");
  return K.getValue();
}

IPAKind AnalyzerOptions::getIPAMode() const {
  auto K = llvm::StringSwitch<llvm::Optional<IPAKind>>(IPAMode)
          .Case("none", IPAK_None)
          .Case("basic-inlining", IPAK_BasicInlining)
          .Case("inlining", IPAK_Inlining)
          .Case("dynamic", IPAK_DynamicDispatch)
          .Case("dynamic-bifurcate", IPAK_DynamicDispatchBifurcate)
          .Default(None);
  assert(K.hasValue() && "IPA Mode is invalid.");

  return K.getValue();
}

bool
AnalyzerOptions::mayInlineCXXMemberFunction(
                                          CXXInlineableMemberKind Param) const {
  if (getIPAMode() < IPAK_Inlining)
    return false;

  auto K =
    llvm::StringSwitch<llvm::Optional<CXXInlineableMemberKind>>(
                                                          CXXMemberInliningMode)
    .Case("constructors", CIMK_Constructors)
    .Case("destructors", CIMK_Destructors)
    .Case("methods", CIMK_MemberFunctions)
    .Case("none", CIMK_None)
    .Default(None);

  assert(K.hasValue() && "Invalid c++ member function inlining mode.");

  return *K >= Param;
}

StringRef AnalyzerOptions::getCheckerStringOption(StringRef CheckerName,
                                                  StringRef OptionName,
                                                  StringRef DefaultVal,
                                                  bool SearchInParents ) const {
  assert(!CheckerName.empty() &&
         "Empty checker name! Make sure the checker object (including it's "
         "bases!) if fully initialized before calling this function!");

  ConfigTable::const_iterator E = Config.end();
  do {
    ConfigTable::const_iterator I =
        Config.find((Twine(CheckerName) + ":" + OptionName).str());
    if (I != E)
      return StringRef(I->getValue());
    size_t Pos = CheckerName.rfind('.');
    if (Pos == StringRef::npos)
      return DefaultVal;
    CheckerName = CheckerName.substr(0, Pos);
  } while (!CheckerName.empty() && SearchInParents);
  return DefaultVal;
}

StringRef AnalyzerOptions::getCheckerStringOption(const ento::CheckerBase *C,
                                                  StringRef OptionName,
                                                  StringRef DefaultVal,
                                                  bool SearchInParents ) const {
  return getCheckerStringOption(
             C->getTagDescription(), OptionName, DefaultVal, SearchInParents);
}

bool AnalyzerOptions::getCheckerBooleanOption(StringRef CheckerName,
                                              StringRef OptionName,
                                              bool DefaultVal,
                                              bool SearchInParents ) const {
  // FIXME: We should emit a warning here if the value is something other than
  // "true", "false", or the empty string (meaning the default value),
  // but the AnalyzerOptions doesn't have access to a diagnostic engine.
  return llvm::StringSwitch<bool>(
      getCheckerStringOption(CheckerName, OptionName,
                             DefaultVal ? "true" : "false",
                             SearchInParents))
      .Case("true", true)
      .Case("false", false)
      .Default(DefaultVal);
}

bool AnalyzerOptions::getCheckerBooleanOption(const ento::CheckerBase *C,
                                              StringRef OptionName,
                                              bool DefaultVal,
                                              bool SearchInParents ) const {
  return getCheckerBooleanOption(
             C->getTagDescription(), OptionName, DefaultVal, SearchInParents);
}

int AnalyzerOptions::getCheckerIntegerOption(StringRef CheckerName,
                                             StringRef OptionName,
                                             int DefaultVal,
                                             bool SearchInParents ) const {
  int Ret = DefaultVal;
  bool HasFailed = getCheckerStringOption(CheckerName, OptionName,
                                          std::to_string(DefaultVal),
                                          SearchInParents)
                     .getAsInteger(10, Ret);
  assert(!HasFailed && "analyzer-config option should be numeric");
  (void)HasFailed;
  return Ret;
}

int AnalyzerOptions::getCheckerIntegerOption(const ento::CheckerBase *C,
                                             StringRef OptionName,
                                             int DefaultVal,
                                             bool SearchInParents ) const {
  return getCheckerIntegerOption(
             C->getTagDescription(), OptionName, DefaultVal, SearchInParents);
}
