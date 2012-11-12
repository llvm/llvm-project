//===--- AnalyzerOptions.h - Analysis Engine Options ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines various options for the static analyzer that are set
// by the frontend and are consulted throughout the analyzer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYZEROPTIONS_H
#define LLVM_CLANG_ANALYZEROPTIONS_H

#include <string>
#include <vector>
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"

namespace clang {
class ASTConsumer;
class DiagnosticsEngine;
class Preprocessor;
class LangOptions;

/// Analysis - Set of available source code analyses.
enum Analyses {
#define ANALYSIS(NAME, CMDFLAG, DESC, SCOPE) NAME,
#include "clang/StaticAnalyzer/Core/Analyses.def"
NumAnalyses
};

/// AnalysisStores - Set of available analysis store models.
enum AnalysisStores {
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) NAME##Model,
#include "clang/StaticAnalyzer/Core/Analyses.def"
NumStores
};

/// AnalysisConstraints - Set of available constraint models.
enum AnalysisConstraints {
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) NAME##Model,
#include "clang/StaticAnalyzer/Core/Analyses.def"
NumConstraints
};

/// AnalysisDiagClients - Set of available diagnostic clients for rendering
///  analysis results.
enum AnalysisDiagClients {
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN, AUTOCREAT) PD_##NAME,
#include "clang/StaticAnalyzer/Core/Analyses.def"
NUM_ANALYSIS_DIAG_CLIENTS
};

/// AnalysisPurgeModes - Set of available strategies for dead symbol removal.
enum AnalysisPurgeMode {
#define ANALYSIS_PURGE(NAME, CMDFLAG, DESC) NAME,
#include "clang/StaticAnalyzer/Core/Analyses.def"
NumPurgeModes
};

/// AnalysisIPAMode - Set of inter-procedural modes.
enum AnalysisIPAMode {
#define ANALYSIS_IPA(NAME, CMDFLAG, DESC) NAME,
#include "clang/StaticAnalyzer/Core/Analyses.def"
NumIPAModes
};

/// AnalysisInlineFunctionSelection - Set of inlining function selection heuristics.
enum AnalysisInliningMode {
#define ANALYSIS_INLINING_MODE(NAME, CMDFLAG, DESC) NAME,
#include "clang/StaticAnalyzer/Core/Analyses.def"
NumInliningModes
};

/// \brief Describes the different kinds of C++ member functions which can be
/// considered for inlining by the analyzer.
///
/// These options are cumulative; enabling one kind of member function will
/// enable all kinds with lower enum values.
enum CXXInlineableMemberKind {
  // Uninitialized = 0,

  /// A dummy mode in which no C++ inlining is enabled.
  CIMK_None = 1,

  /// Refers to regular member function and operator calls.
  CIMK_MemberFunctions,

  /// Refers to constructors (implicit or explicit).
  ///
  /// Note that a constructor will not be inlined if the corresponding
  /// destructor is non-trivial.
  CIMK_Constructors,

  /// Refers to destructors (implicit or explicit).
  CIMK_Destructors
};


class AnalyzerOptions : public llvm::RefCountedBase<AnalyzerOptions> {
public:
  typedef llvm::StringMap<std::string> ConfigTable;

  /// \brief Pair of checker name and enable/disable.
  std::vector<std::pair<std::string, bool> > CheckersControlList;
  
  /// \brief A key-value table of use-specified configuration values.
  ConfigTable Config;
  AnalysisStores AnalysisStoreOpt;
  AnalysisConstraints AnalysisConstraintsOpt;
  AnalysisDiagClients AnalysisDiagOpt;
  AnalysisPurgeMode AnalysisPurgeOpt;
  
  // \brief The interprocedural analysis mode.
  AnalysisIPAMode IPAMode;
  
  std::string AnalyzeSpecificFunction;
  
  /// \brief The maximum number of exploded nodes the analyzer will generate.
  unsigned MaxNodes;
  
  /// \brief The maximum number of times the analyzer visits a block.
  unsigned maxBlockVisitOnPath;
  
  
  unsigned ShowCheckerHelp : 1;
  unsigned AnalyzeAll : 1;
  unsigned AnalyzerDisplayProgress : 1;
  unsigned AnalyzeNestedBlocks : 1;
  
  /// \brief The flag regulates if we should eagerly assume evaluations of
  /// conditionals, thus, bifurcating the path.
  ///
  /// This flag indicates how the engine should handle expressions such as: 'x =
  /// (y != 0)'.  When this flag is true then the subexpression 'y != 0' will be
  /// eagerly assumed to be true or false, thus evaluating it to the integers 0
  /// or 1 respectively.  The upside is that this can increase analysis
  /// precision until we have a better way to lazily evaluate such logic.  The
  /// downside is that it eagerly bifurcates paths.
  unsigned eagerlyAssumeBinOpBifurcation : 1;
  
  unsigned TrimGraph : 1;
  unsigned visualizeExplodedGraphWithGraphViz : 1;
  unsigned visualizeExplodedGraphWithUbiGraph : 1;
  unsigned UnoptimizedCFG : 1;
  unsigned PrintStats : 1;
  
  /// \brief Do not re-analyze paths leading to exhausted nodes with a different
  /// strategy. We get better code coverage when retry is enabled.
  unsigned NoRetryExhausted : 1;
  
  /// \brief The inlining stack depth limit.
  unsigned InlineMaxStackDepth;
  
  /// \brief The mode of function selection used during inlining.
  unsigned InlineMaxFunctionSize;

  /// \brief The mode of function selection used during inlining.
  AnalysisInliningMode InliningMode;

private:
  /// Controls which C++ member functions will be considered for inlining.
  CXXInlineableMemberKind CXXMemberInliningMode;
  
  /// \sa includeTemporaryDtorsInCFG
  llvm::Optional<bool> IncludeTemporaryDtorsInCFG;
  
  /// \sa mayInlineCXXStandardLibrary
  llvm::Optional<bool> InlineCXXStandardLibrary;
  
  /// \sa mayInlineTemplateFunctions
  llvm::Optional<bool> InlineTemplateFunctions;

  /// \sa mayInlineObjCMethod
  llvm::Optional<bool> ObjCInliningMode;

  // Cache of the "ipa-always-inline-size" setting.
  // \sa getAlwaysInlineSize
  llvm::Optional<unsigned> AlwaysInlineSize;

  /// \sa shouldPruneNullReturnPaths
  llvm::Optional<bool> PruneNullReturnPaths;

  /// \sa shouldAvoidSuppressingNullArgumentPaths
  llvm::Optional<bool> AvoidSuppressingNullArgumentPaths;
  
  /// \sa getGraphTrimInterval
  llvm::Optional<unsigned> GraphTrimInterval;

  /// Interprets an option's string value as a boolean.
  ///
  /// Accepts the strings "true" and "false".
  /// If an option value is not provided, returns the given \p DefaultVal.
  bool getBooleanOption(StringRef Name, bool DefaultVal);

  /// Variant that accepts a Optional value to cache the result.
  bool getBooleanOption(llvm::Optional<bool> &V, StringRef Name,
                        bool DefaultVal);
  
  /// Interprets an option's string value as an integer value.
  int getOptionAsInteger(llvm::StringRef Name, int DefaultVal);

public:
  /// Returns the option controlling which C++ member functions will be
  /// considered for inlining.
  ///
  /// This is controlled by the 'c++-inlining' config option.
  ///
  /// \sa CXXMemberInliningMode
  bool mayInlineCXXMemberFunction(CXXInlineableMemberKind K);

  /// Returns true if ObjectiveC inlining is enabled, false otherwise.
  bool mayInlineObjCMethod();

  /// Returns whether or not the destructors for C++ temporary objects should
  /// be included in the CFG.
  ///
  /// This is controlled by the 'cfg-temporary-dtors' config option, which
  /// accepts the values "true" and "false".
  bool includeTemporaryDtorsInCFG();

  /// Returns whether or not C++ standard library functions may be considered
  /// for inlining.
  ///
  /// This is controlled by the 'c++-stdlib-inlining' config option, which
  /// accepts the values "true" and "false".
  bool mayInlineCXXStandardLibrary();

  /// Returns whether or not templated functions may be considered for inlining.
  ///
  /// This is controlled by the 'c++-template-inlining' config option, which
  /// accepts the values "true" and "false".
  bool mayInlineTemplateFunctions();

  /// Returns whether or not paths that go through null returns should be
  /// suppressed.
  ///
  /// This is a heuristic for avoiding bug reports with paths that go through
  /// inlined functions that are more defensive than their callers.
  ///
  /// This is controlled by the 'suppress-null-return-paths' config option,
  /// which accepts the values "true" and "false".
  bool shouldPruneNullReturnPaths();

  /// Returns whether a bug report should \em not be suppressed if its path
  /// includes a call with a null argument, even if that call has a null return.
  ///
  /// This option has no effect when #shouldPruneNullReturnPaths() is false.
  ///
  /// This is a counter-heuristic to avoid false negatives.
  ///
  /// This is controlled by the 'avoid-suppressing-null-argument-paths' config
  /// option, which accepts the values "true" and "false".
  bool shouldAvoidSuppressingNullArgumentPaths();

  // Returns the size of the functions (in basic blocks), which should be
  // considered to be small enough to always inline.
  //
  // This is controlled by "ipa-always-inline-size" analyzer-config option.
  unsigned getAlwaysInlineSize();
  
  /// Returns true if the analyzer engine should synthesize fake bodies
  /// for well-known functions.
  bool shouldSynthesizeBodies();

  /// Returns how often nodes in the ExplodedGraph should be recycled to save
  /// memory.
  ///
  /// This is controlled by the 'graph-trim-interval' config option. To disable
  /// node reclamation, set the option to "0".
  unsigned getGraphTrimInterval();

public:
  AnalyzerOptions() : CXXMemberInliningMode() {
    AnalysisStoreOpt = RegionStoreModel;
    AnalysisConstraintsOpt = RangeConstraintsModel;
    AnalysisDiagOpt = PD_HTML;
    AnalysisPurgeOpt = PurgeStmt;
    IPAMode = DynamicDispatchBifurcate;
    ShowCheckerHelp = 0;
    AnalyzeAll = 0;
    AnalyzerDisplayProgress = 0;
    AnalyzeNestedBlocks = 0;
    eagerlyAssumeBinOpBifurcation = 0;
    TrimGraph = 0;
    visualizeExplodedGraphWithGraphViz = 0;
    visualizeExplodedGraphWithUbiGraph = 0;
    UnoptimizedCFG = 0;
    PrintStats = 0;
    NoRetryExhausted = 0;
    // Cap the stack depth at 4 calls (5 stack frames, base + 4 calls).
    InlineMaxStackDepth = 5;
    InlineMaxFunctionSize = 200;
    InliningMode = NoRedundancy;
  }
};
  
typedef llvm::IntrusiveRefCntPtr<AnalyzerOptions> AnalyzerOptionsRef;
  
}

#endif
