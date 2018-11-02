//===- AnalyzerOptions.cpp - Analysis Engine Options ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#define CHECKER(FULLNAME, CLASS, DESCFILE, HELPTEXT, GROUPINDEX, HIDDEN)       \
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

UserModeKind AnalyzerOptions::getUserMode() {
  if (!UserMode.hasValue()) {
    UserMode = getOptionAsString("mode", "deep");
  }

  auto K = llvm::StringSwitch<llvm::Optional<UserModeKind>>(*UserMode)
    .Case("shallow", UMK_Shallow)
    .Case("deep", UMK_Deep)
    .Default(None);
  assert(UserMode.hasValue() && "User mode is invalid.");
  return K.getValue();
}

ExplorationStrategyKind
AnalyzerOptions::getExplorationStrategy() {
  if (!ExplorationStrategy.hasValue()) {
    ExplorationStrategy = getOptionAsString("exploration_strategy",
                                            "unexplored_first_queue");
  }
  auto K =
    llvm::StringSwitch<llvm::Optional<ExplorationStrategyKind>>(
                                                           *ExplorationStrategy)
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

IPAKind AnalyzerOptions::getIPAMode() {
  if (!IPAMode.hasValue()) {
    switch (getUserMode()) {
    case UMK_Shallow:
      IPAMode = getOptionAsString("ipa", "inlining");
      break;
    case UMK_Deep:
      IPAMode = getOptionAsString("ipa", "dynamic-bifurcate");
      break;
    }
  }
  auto K = llvm::StringSwitch<llvm::Optional<IPAKind>>(*IPAMode)
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
AnalyzerOptions::mayInlineCXXMemberFunction(CXXInlineableMemberKind Param) {
  if (!CXXMemberInliningMode.hasValue()) {
    CXXMemberInliningMode = getOptionAsString("c++-inlining", "destructors");
  }

  if (getIPAMode() < IPAK_Inlining)
    return false;

  auto K =
    llvm::StringSwitch<llvm::Optional<CXXInlineableMemberKind>>(
                                                         *CXXMemberInliningMode)
    .Case("constructors", CIMK_Constructors)
    .Case("destructors", CIMK_Destructors)
    .Case("methods", CIMK_MemberFunctions)
    .Case("none", CIMK_None)
    .Default(None);

  assert(K.hasValue() && "Invalid c++ member function inlining mode.");

  return *K >= Param;
}

static StringRef toString(bool b) { return b ? "true" : "false"; }

StringRef AnalyzerOptions::getCheckerOption(StringRef CheckerName,
                                            StringRef OptionName,
                                            StringRef Default,
                                            bool SearchInParents) {
  // Search for a package option if the option for the checker is not specified
  // and search in parents is enabled.
  ConfigTable::const_iterator E = Config.end();
  do {
    ConfigTable::const_iterator I =
        Config.find((Twine(CheckerName) + ":" + OptionName).str());
    if (I != E)
      return StringRef(I->getValue());
    size_t Pos = CheckerName.rfind('.');
    if (Pos == StringRef::npos)
      return Default;
    CheckerName = CheckerName.substr(0, Pos);
  } while (!CheckerName.empty() && SearchInParents);
  return Default;
}

bool AnalyzerOptions::getBooleanOption(StringRef Name, bool DefaultVal,
                                       const CheckerBase *C,
                                       bool SearchInParents) {
  // FIXME: We should emit a warning here if the value is something other than
  // "true", "false", or the empty string (meaning the default value),
  // but the AnalyzerOptions doesn't have access to a diagnostic engine.
  StringRef Default = toString(DefaultVal);
  StringRef V =
      C ? getCheckerOption(C->getTagDescription(), Name, Default,
                           SearchInParents)
        : getOptionAsString(Name, Default);
  return llvm::StringSwitch<bool>(V)
      .Case("true", true)
      .Case("false", false)
      .Default(DefaultVal);
}

bool AnalyzerOptions::getBooleanOption(Optional<bool> &V, StringRef Name,
                                       bool DefaultVal, const CheckerBase *C,
                                       bool SearchInParents) {
  if (!V.hasValue())
    V = getBooleanOption(Name, DefaultVal, C, SearchInParents);
  return V.getValue();
}

int AnalyzerOptions::getOptionAsInteger(StringRef Name, int DefaultVal,
                                        const CheckerBase *C,
                                        bool SearchInParents) {
  SmallString<10> StrBuf;
  llvm::raw_svector_ostream OS(StrBuf);
  OS << DefaultVal;

  StringRef V = C ? getCheckerOption(C->getTagDescription(), Name, OS.str(),
                                     SearchInParents)
                  : getOptionAsString(Name, OS.str());

  int Res = DefaultVal;
  bool b = V.getAsInteger(10, Res);
  assert(!b && "analyzer-config option should be numeric");
  (void)b;
  return Res;
}

unsigned AnalyzerOptions::getOptionAsUInt(Optional<unsigned> &V, StringRef Name,
                                          unsigned DefaultVal,
                                          const CheckerBase *C,
                                          bool SearchInParents) {
  if (!V.hasValue())
    V = getOptionAsInteger(Name, DefaultVal, C, SearchInParents);
  return V.getValue();
}

StringRef AnalyzerOptions::getOptionAsString(StringRef Name,
                                             StringRef DefaultVal,
                                             const CheckerBase *C,
                                             bool SearchInParents) {
  return C ? getCheckerOption(C->getTagDescription(), Name, DefaultVal,
                              SearchInParents)
           : StringRef(
                 Config.insert(std::make_pair(Name, DefaultVal)).first->second);
}

StringRef AnalyzerOptions::getOptionAsString(Optional<StringRef> &V,
                                             StringRef Name,
                                             StringRef DefaultVal,
                                             const ento::CheckerBase *C,
                                             bool SearchInParents) {
  if (!V.hasValue())
    V = getOptionAsString(Name, DefaultVal, C, SearchInParents);
  return V.getValue();
}

static bool getOption(AnalyzerOptions &A, Optional<bool> &V, StringRef Name,
                      bool DefaultVal) {
  return A.getBooleanOption(V, Name, DefaultVal);
}

static unsigned getOption(AnalyzerOptions &A, Optional<unsigned> &V,
                          StringRef Name, unsigned DefaultVal) {
  return A.getOptionAsUInt(V, Name, DefaultVal);
}

static StringRef getOption(AnalyzerOptions &A, Optional<StringRef> &V,
                           StringRef Name, StringRef DefaultVal) {
  return A.getOptionAsString(V, Name, DefaultVal);
}

#define ANALYZER_OPTION_GEN_FN(TYPE, NAME, CMDFLAG, DESC, DEFAULT_VAL,  \
                                CREATE_FN)                              \
TYPE AnalyzerOptions::CREATE_FN() {                                     \
  return getOption(*this, NAME, CMDFLAG, DEFAULT_VAL);                  \
}

#define ANALYZER_OPTION_GEN_FN_DEPENDS_ON_USER_MODE(                    \
    TYPE, NAME, CMDFLAG, DESC, SHALLOW_VAL, DEEP_VAL, CREATE_FN)        \
TYPE AnalyzerOptions::CREATE_FN() {                                     \
  switch (getUserMode()) {                                              \
  case UMK_Shallow:                                                     \
    return getOption(*this, NAME, CMDFLAG, SHALLOW_VAL);                \
  case UMK_Deep:                                                        \
    return getOption(*this, NAME, CMDFLAG, DEEP_VAL);                   \
  }                                                                     \
                                                                        \
  llvm_unreachable("Unknown usermode!");                                \
  return {};                                                            \
}

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.def"

#undef ANALYZER_OPTION_GEN_FN_DEPENDS_ON_USER_MODE
#undef ANALYZER_OPTION_WITH_FN

StringRef AnalyzerOptions::getCTUDir() {
  if (!CTUDir.hasValue()) {
    CTUDir = getOptionAsString("ctu-dir", "");
    if (!llvm::sys::fs::is_directory(*CTUDir))
      CTUDir = "";
  }
  return CTUDir.getValue();
}
