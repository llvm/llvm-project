//===-- AnalyzerOptions.cpp - Analysis Engine Options -----------*- C++ -*-===//
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

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

AnalyzerOptions::UserModeKind AnalyzerOptions::getUserMode() {
  if (UserMode == UMK_NotSet) {
    StringRef ModeStr =
        Config.insert(std::make_pair("mode", "deep")).first->second;
    UserMode = llvm::StringSwitch<UserModeKind>(ModeStr)
      .Case("shallow", UMK_Shallow)
      .Case("deep", UMK_Deep)
      .Default(UMK_NotSet);
    assert(UserMode != UMK_NotSet && "User mode is invalid.");
  }
  return UserMode;
}

AnalyzerOptions::ExplorationStrategyKind
AnalyzerOptions::getExplorationStrategy() {
  if (ExplorationStrategy == ExplorationStrategyKind::NotSet) {
    StringRef StratStr =
        Config
            .insert(std::make_pair("exploration_strategy", "unexplored_first_queue"))
            .first->second;
    ExplorationStrategy =
        llvm::StringSwitch<ExplorationStrategyKind>(StratStr)
            .Case("dfs", ExplorationStrategyKind::DFS)
            .Case("bfs", ExplorationStrategyKind::BFS)
            .Case("unexplored_first",
                  ExplorationStrategyKind::UnexploredFirst)
            .Case("unexplored_first_queue",
                  ExplorationStrategyKind::UnexploredFirstQueue)
            .Case("bfs_block_dfs_contents",
                  ExplorationStrategyKind::BFSBlockDFSContents)
            .Default(ExplorationStrategyKind::NotSet);
    assert(ExplorationStrategy != ExplorationStrategyKind::NotSet &&
           "User mode is invalid.");
  }
  return ExplorationStrategy;
}

IPAKind AnalyzerOptions::getIPAMode() {
  if (IPAMode == IPAK_NotSet) {

    // Use the User Mode to set the default IPA value.
    // Note, we have to add the string to the Config map for the ConfigDumper
    // checker to function properly.
    const char *DefaultIPA = nullptr;
    UserModeKind HighLevelMode = getUserMode();
    if (HighLevelMode == UMK_Shallow)
      DefaultIPA = "inlining";
    else if (HighLevelMode == UMK_Deep)
      DefaultIPA = "dynamic-bifurcate";
    assert(DefaultIPA);

    // Lookup the ipa configuration option, use the default from User Mode.
    StringRef ModeStr =
        Config.insert(std::make_pair("ipa", DefaultIPA)).first->second;
    IPAKind IPAConfig = llvm::StringSwitch<IPAKind>(ModeStr)
            .Case("none", IPAK_None)
            .Case("basic-inlining", IPAK_BasicInlining)
            .Case("inlining", IPAK_Inlining)
            .Case("dynamic", IPAK_DynamicDispatch)
            .Case("dynamic-bifurcate", IPAK_DynamicDispatchBifurcate)
            .Default(IPAK_NotSet);
    assert(IPAConfig != IPAK_NotSet && "IPA Mode is invalid.");

    // Set the member variable.
    IPAMode = IPAConfig;
  }

  return IPAMode;
}

bool
AnalyzerOptions::mayInlineCXXMemberFunction(CXXInlineableMemberKind K) {
  if (getIPAMode() < IPAK_Inlining)
    return false;

  if (!CXXMemberInliningMode) {
    static const char *ModeKey = "c++-inlining";

    StringRef ModeStr =
        Config.insert(std::make_pair(ModeKey, "destructors")).first->second;

    CXXInlineableMemberKind &MutableMode =
      const_cast<CXXInlineableMemberKind &>(CXXMemberInliningMode);

    MutableMode = llvm::StringSwitch<CXXInlineableMemberKind>(ModeStr)
      .Case("constructors", CIMK_Constructors)
      .Case("destructors", CIMK_Destructors)
      .Case("none", CIMK_None)
      .Case("methods", CIMK_MemberFunctions)
      .Default(CXXInlineableMemberKind());

    if (!MutableMode) {
      // FIXME: We should emit a warning here about an unknown inlining kind,
      // but the AnalyzerOptions doesn't have access to a diagnostic engine.
      MutableMode = CIMK_None;
    }
  }

  return CXXMemberInliningMode >= K;
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
        : StringRef(Config.insert(std::make_pair(Name, Default)).first->second);
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

bool AnalyzerOptions::includeTemporaryDtorsInCFG() {
  return getBooleanOption(IncludeTemporaryDtorsInCFG,
                          "cfg-temporary-dtors",
                          /* Default = */ true);
}

bool AnalyzerOptions::includeImplicitDtorsInCFG() {
  return getBooleanOption(IncludeImplicitDtorsInCFG,
                          "cfg-implicit-dtors",
                          /* Default = */ true);
}

bool AnalyzerOptions::includeLifetimeInCFG() {
  return getBooleanOption(IncludeLifetimeInCFG, "cfg-lifetime",
                          /* Default = */ false);
}

bool AnalyzerOptions::includeLoopExitInCFG() {
  return getBooleanOption(IncludeLoopExitInCFG, "cfg-loopexit",
                          /* Default = */ false);
}

bool AnalyzerOptions::includeRichConstructorsInCFG() {
  return getBooleanOption(IncludeRichConstructorsInCFG,
                          "cfg-rich-constructors",
                          /* Default = */ true);
}

bool AnalyzerOptions::mayInlineCXXStandardLibrary() {
  return getBooleanOption(InlineCXXStandardLibrary,
                          "c++-stdlib-inlining",
                          /*Default=*/true);
}

bool AnalyzerOptions::mayInlineTemplateFunctions() {
  return getBooleanOption(InlineTemplateFunctions,
                          "c++-template-inlining",
                          /*Default=*/true);
}

bool AnalyzerOptions::mayInlineCXXAllocator() {
  return getBooleanOption(InlineCXXAllocator,
                          "c++-allocator-inlining",
                          /*Default=*/true);
}

bool AnalyzerOptions::mayInlineCXXContainerMethods() {
  return getBooleanOption(InlineCXXContainerMethods,
                          "c++-container-inlining",
                          /*Default=*/false);
}

bool AnalyzerOptions::mayInlineCXXSharedPtrDtor() {
  return getBooleanOption(InlineCXXSharedPtrDtor,
                          "c++-shared_ptr-inlining",
                          /*Default=*/false);
}

bool AnalyzerOptions::mayInlineCXXTemporaryDtors() {
  return getBooleanOption(InlineCXXTemporaryDtors,
                          "c++-temp-dtor-inlining",
                          /*Default=*/false);
}

bool AnalyzerOptions::mayInlineObjCMethod() {
  return getBooleanOption(ObjCInliningMode,
                          "objc-inlining",
                          /* Default = */ true);
}

bool AnalyzerOptions::shouldSuppressNullReturnPaths() {
  return getBooleanOption(SuppressNullReturnPaths,
                          "suppress-null-return-paths",
                          /* Default = */ true);
}

bool AnalyzerOptions::shouldAvoidSuppressingNullArgumentPaths() {
  return getBooleanOption(AvoidSuppressingNullArgumentPaths,
                          "avoid-suppressing-null-argument-paths",
                          /* Default = */ false);
}

bool AnalyzerOptions::shouldSuppressInlinedDefensiveChecks() {
  return getBooleanOption(SuppressInlinedDefensiveChecks,
                          "suppress-inlined-defensive-checks",
                          /* Default = */ true);
}

bool AnalyzerOptions::shouldSuppressFromCXXStandardLibrary() {
  return getBooleanOption(SuppressFromCXXStandardLibrary,
                          "suppress-c++-stdlib",
                          /* Default = */ true);
}

bool AnalyzerOptions::shouldReportIssuesInMainSourceFile() {
  return getBooleanOption(ReportIssuesInMainSourceFile,
                          "report-in-main-source-file",
                          /* Default = */ false);
}


bool AnalyzerOptions::shouldWriteStableReportFilename() {
  return getBooleanOption(StableReportFilename,
                          "stable-report-filename",
                          /* Default = */ false);
}

bool AnalyzerOptions::shouldSerializeStats() {
  return getBooleanOption(SerializeStats,
                          "serialize-stats",
                          /* Default = */ false);
}

int AnalyzerOptions::getOptionAsInteger(StringRef Name, int DefaultVal,
                                        const CheckerBase *C,
                                        bool SearchInParents) {
  SmallString<10> StrBuf;
  llvm::raw_svector_ostream OS(StrBuf);
  OS << DefaultVal;

  StringRef V = C ? getCheckerOption(C->getTagDescription(), Name, OS.str(),
                                     SearchInParents)
                  : StringRef(Config.insert(std::make_pair(Name, OS.str()))
                                  .first->second);

  int Res = DefaultVal;
  bool b = V.getAsInteger(10, Res);
  assert(!b && "analyzer-config option should be numeric");
  (void)b;
  return Res;
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

unsigned AnalyzerOptions::getAlwaysInlineSize() {
  if (!AlwaysInlineSize.hasValue())
    AlwaysInlineSize = getOptionAsInteger("ipa-always-inline-size", 3);
  return AlwaysInlineSize.getValue();
}

unsigned AnalyzerOptions::getMaxInlinableSize() {
  if (!MaxInlinableSize.hasValue()) {

    int DefaultValue = 0;
    UserModeKind HighLevelMode = getUserMode();
    switch (HighLevelMode) {
      default:
        llvm_unreachable("Invalid mode.");
      case UMK_Shallow:
        DefaultValue = 4;
        break;
      case UMK_Deep:
        DefaultValue = 100;
        break;
    }

    MaxInlinableSize = getOptionAsInteger("max-inlinable-size", DefaultValue);
  }
  return MaxInlinableSize.getValue();
}

unsigned AnalyzerOptions::getGraphTrimInterval() {
  if (!GraphTrimInterval.hasValue())
    GraphTrimInterval = getOptionAsInteger("graph-trim-interval", 1000);
  return GraphTrimInterval.getValue();
}

unsigned AnalyzerOptions::getMaxTimesInlineLarge() {
  if (!MaxTimesInlineLarge.hasValue())
    MaxTimesInlineLarge = getOptionAsInteger("max-times-inline-large", 32);
  return MaxTimesInlineLarge.getValue();
}

unsigned AnalyzerOptions::getMinCFGSizeTreatFunctionsAsLarge() {
  if (!MinCFGSizeTreatFunctionsAsLarge.hasValue())
    MinCFGSizeTreatFunctionsAsLarge = getOptionAsInteger(
      "min-cfg-size-treat-functions-as-large", 14);
  return MinCFGSizeTreatFunctionsAsLarge.getValue();
}

unsigned AnalyzerOptions::getMaxNodesPerTopLevelFunction() {
  if (!MaxNodesPerTopLevelFunction.hasValue()) {
    int DefaultValue = 0;
    UserModeKind HighLevelMode = getUserMode();
    switch (HighLevelMode) {
      default:
        llvm_unreachable("Invalid mode.");
      case UMK_Shallow:
        DefaultValue = 75000;
        break;
      case UMK_Deep:
        DefaultValue = 225000;
        break;
    }
    MaxNodesPerTopLevelFunction = getOptionAsInteger("max-nodes", DefaultValue);
  }
  return MaxNodesPerTopLevelFunction.getValue();
}

bool AnalyzerOptions::shouldSynthesizeBodies() {
  return getBooleanOption("faux-bodies", true);
}

bool AnalyzerOptions::shouldPrunePaths() {
  return getBooleanOption("prune-paths", true);
}

bool AnalyzerOptions::shouldConditionalizeStaticInitializers() {
  return getBooleanOption("cfg-conditional-static-initializers", true);
}

bool AnalyzerOptions::shouldInlineLambdas() {
  if (!InlineLambdas.hasValue())
    InlineLambdas = getBooleanOption("inline-lambdas", /*Default=*/true);
  return InlineLambdas.getValue();
}

bool AnalyzerOptions::shouldWidenLoops() {
  if (!WidenLoops.hasValue())
    WidenLoops = getBooleanOption("widen-loops", /*Default=*/false);
  return WidenLoops.getValue();
}

bool AnalyzerOptions::shouldUnrollLoops() {
  if (!UnrollLoops.hasValue())
    UnrollLoops = getBooleanOption("unroll-loops", /*Default=*/false);
  return UnrollLoops.getValue();
}

bool AnalyzerOptions::shouldDisplayNotesAsEvents() {
  if (!DisplayNotesAsEvents.hasValue())
    DisplayNotesAsEvents =
        getBooleanOption("notes-as-events", /*Default=*/false);
  return DisplayNotesAsEvents.getValue();
}

StringRef AnalyzerOptions::getCTUDir() {
  if (!CTUDir.hasValue()) {
    CTUDir = getOptionAsString("ctu-dir", "");
    if (!llvm::sys::fs::is_directory(*CTUDir))
      CTUDir = "";
  }
  return CTUDir.getValue();
}

bool AnalyzerOptions::naiveCTUEnabled() {
  if (!NaiveCTU.hasValue()) {
    NaiveCTU = getBooleanOption("experimental-enable-naive-ctu-analysis",
                                /*Default=*/false);
  }
  return NaiveCTU.getValue();
}

StringRef AnalyzerOptions::getCTUIndexName() {
  if (!CTUIndexName.hasValue())
    CTUIndexName = getOptionAsString("ctu-index-name", "externalFnMap.txt");
  return CTUIndexName.getValue();
}
