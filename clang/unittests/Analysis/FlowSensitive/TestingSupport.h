//===--- TestingSupport.h - Testing utils for dataflow analyses -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to simplify testing of dataflow analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_TESTING_SUPPORT_H_
#define LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_TESTING_SUPPORT_H_

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/LLVM.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"

namespace clang {
namespace dataflow {

// Requires a `<<` operator for the `Lattice` type.
// FIXME: move to a non-test utility library.
template <typename Lattice>
std::ostream &operator<<(std::ostream &OS,
                         const DataflowAnalysisState<Lattice> &S) {
  // FIXME: add printing support for the environment.
  return OS << "{lattice=" << S.Lattice << ", environment=...}";
}

namespace test {

/// Returns the environment at the program point marked with `Annotation` from
/// the mapping of annotated program points to analysis state.
///
/// Requirements:
///
///   `Annotation` must be present as a key in `AnnotationStates`.
template <typename LatticeT>
const Environment &getEnvironmentAtAnnotation(
    const llvm::StringMap<DataflowAnalysisState<LatticeT>> &AnnotationStates,
    llvm::StringRef Annotation) {
  auto It = AnnotationStates.find(Annotation);
  assert(It != AnnotationStates.end());
  return It->getValue().Env;
}

/// Contains data structures required and produced by a dataflow analysis run.
struct AnalysisOutputs {
  /// Input code that is analyzed. Points within the code may be marked with
  /// annotations to facilitate testing.
  ///
  /// Example:
  /// void target(int *x) {
  ///   *x; // [[p]]
  /// }
  /// From the annotation `p`, the line number and analysis state immediately
  /// after the statement `*x` can be retrieved and verified.
  llvm::Annotations Code;
  /// AST context generated from `Code`.
  ASTContext &ASTCtx;
  /// The function whose body is analyzed.
  const FunctionDecl *Target;
  /// Contains the control flow graph built from the body of the `Target`
  /// function and is analyzed.
  const ControlFlowContext &CFCtx;
  /// The analysis to be run.
  TypeErasedDataflowAnalysis &Analysis;
  /// Initial state to start the analysis.
  const Environment &InitEnv;
  // Stores the state of a CFG block if it has been evaluated by the analysis.
  // The indices correspond to the block IDs.
  llvm::ArrayRef<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates;
};

/// Arguments for building the dataflow analysis.
template <typename AnalysisT> struct AnalysisInputs {
  /// Required fields are set in constructor.
  AnalysisInputs(
      llvm::StringRef CodeArg,
      ast_matchers::internal::Matcher<FunctionDecl> TargetFuncMatcherArg,
      std::function<AnalysisT(ASTContext &, Environment &)> MakeAnalysisArg)
      : Code(CodeArg), TargetFuncMatcher(std::move(TargetFuncMatcherArg)),
        MakeAnalysis(std::move(MakeAnalysisArg)) {}

  /// Optional fields can be set with methods of the form `withFieldName(...)`.
  AnalysisInputs<AnalysisT> &&
  withSetupTest(std::function<llvm::Error(AnalysisOutputs &)> Arg) && {
    SetupTest = std::move(Arg);
    return std::move(*this);
  }
  AnalysisInputs<AnalysisT> &&withPostVisitCFG(
      std::function<void(
          ASTContext &, const CFGElement &,
          const TransferStateForDiagnostics<typename AnalysisT::Lattice> &)>
          Arg) && {
    PostVisitCFG = std::move(Arg);
    return std::move(*this);
  }
  AnalysisInputs<AnalysisT> &&withASTBuildArgs(ArrayRef<std::string> Arg) && {
    ASTBuildArgs = std::move(Arg);
    return std::move(*this);
  }
  AnalysisInputs<AnalysisT> &&
  withASTBuildVirtualMappedFiles(tooling::FileContentMappings Arg) && {
    ASTBuildVirtualMappedFiles = std::move(Arg);
    return std::move(*this);
  }

  /// Required. Input code that is analyzed.
  llvm::StringRef Code;
  /// Required. The body of the function which matches this matcher is analyzed.
  ast_matchers::internal::Matcher<FunctionDecl> TargetFuncMatcher;
  /// Required. The analysis to be run is constructed with this function that
  /// takes as argument the AST generated from the code being analyzed and the
  /// initial state from which the analysis starts with.
  std::function<AnalysisT(ASTContext &, Environment &)> MakeAnalysis;
  /// Optional. If provided, this function is executed immediately before
  /// running the dataflow analysis to allow for additional setup. All fields in
  /// the `AnalysisOutputs` argument will be initialized except for the
  /// `BlockStates` field which is only computed later during the analysis.
  std::function<llvm::Error(AnalysisOutputs &)> SetupTest = nullptr;
  /// Optional. If provided, this function is applied on each CFG element after
  /// the analysis has been run.
  std::function<void(
      ASTContext &, const CFGElement &,
      const TransferStateForDiagnostics<typename AnalysisT::Lattice> &)>
      PostVisitCFG = nullptr;

  /// Optional. Options for building the AST context.
  ArrayRef<std::string> ASTBuildArgs = {};
  /// Optional. Options for building the AST context.
  tooling::FileContentMappings ASTBuildVirtualMappedFiles = {};
};

/// Returns assertions based on annotations that are present after statements in
/// `AnnotatedCode`.
llvm::Expected<llvm::DenseMap<const Stmt *, std::string>>
buildStatementToAnnotationMapping(const FunctionDecl *Func,
                                  llvm::Annotations AnnotatedCode);

/// Returns line numbers and content of the annotations in `AnnotatedCode`.
llvm::DenseMap<unsigned, std::string>
buildLineToAnnotationMapping(SourceManager &SM,
                             llvm::Annotations AnnotatedCode);

/// Runs dataflow specified from `AI.MakeAnalysis` and `AI.PostVisitCFG` on the
/// body of the function that matches `AI.TargetFuncMatcher` in `AI.Code`.
/// Given the analysis outputs, `VerifyResults` checks that the results from the
/// analysis are correct.
///
/// Requirements:
///
///   `AnalysisT` contains a type `Lattice`.
///
///   `Code`, `TargetFuncMatcher` and `MakeAnalysis` must be provided in `AI`.
///
///   `VerifyResults` must be provided.
template <typename AnalysisT>
llvm::Error
checkDataflow(AnalysisInputs<AnalysisT> AI,
              std::function<void(const AnalysisOutputs &)> VerifyResults) {
  // Build AST context from code.
  llvm::Annotations AnnotatedCode(AI.Code);
  auto Unit = tooling::buildASTFromCodeWithArgs(
      AnnotatedCode.code(), AI.ASTBuildArgs, "input.cc", "clang-dataflow-test",
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(),
      AI.ASTBuildVirtualMappedFiles);
  auto &Context = Unit->getASTContext();

  if (Context.getDiagnostics().getClient()->getNumErrors() != 0) {
    return llvm::make_error<llvm::StringError>(
        llvm::errc::invalid_argument, "Source file has syntax or type errors, "
                                      "they were printed to the test log");
  }

  // Get AST node of target function.
  const FunctionDecl *Target = ast_matchers::selectFirst<FunctionDecl>(
      "target", ast_matchers::match(
                    ast_matchers::functionDecl(ast_matchers::isDefinition(),
                                               AI.TargetFuncMatcher)
                        .bind("target"),
                    Context));
  if (Target == nullptr)
    return llvm::make_error<llvm::StringError>(
        llvm::errc::invalid_argument, "Could not find target function.");

  // Build control flow graph from body of target function.
  auto MaybeCFCtx =
      ControlFlowContext::build(Target, *Target->getBody(), Context);
  if (!MaybeCFCtx)
    return MaybeCFCtx.takeError();
  auto &CFCtx = *MaybeCFCtx;

  // Initialize states for running dataflow analysis.
  DataflowAnalysisContext DACtx(std::make_unique<WatchedLiteralsSolver>());
  Environment InitEnv(DACtx, *Target);
  auto Analysis = AI.MakeAnalysis(Context, InitEnv);
  std::function<void(const CFGElement &,
                     const TypeErasedDataflowAnalysisState &)>
      PostVisitCFGClosure = nullptr;
  if (AI.PostVisitCFG) {
    PostVisitCFGClosure = [&AI, &Context](
                              const CFGElement &Element,
                              const TypeErasedDataflowAnalysisState &State) {
      AI.PostVisitCFG(Context, Element,
                      TransferStateForDiagnostics<typename AnalysisT::Lattice>(
                          llvm::any_cast<const typename AnalysisT::Lattice &>(
                              State.Lattice.Value),
                          State.Env));
    };
  }

  // Additional test setup.
  AnalysisOutputs AO{AnnotatedCode, Context, Target, CFCtx,
                     Analysis,      InitEnv, {}};
  if (AI.SetupTest) {
    if (auto Error = AI.SetupTest(AO))
      return Error;
  }

  // If successful, the dataflow analysis returns a mapping from block IDs to
  // the post-analysis states for the CFG blocks that have been evaluated.
  llvm::Expected<std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>>
      MaybeBlockStates = runTypeErasedDataflowAnalysis(CFCtx, Analysis, InitEnv,
                                                       PostVisitCFGClosure);
  if (!MaybeBlockStates)
    return MaybeBlockStates.takeError();
  AO.BlockStates = *MaybeBlockStates;

  // Verify dataflow analysis outputs.
  VerifyResults(AO);
  return llvm::Error::success();
}

/// Runs dataflow specified from `AI.MakeAnalysis` and `AI.PostVisitCFG` on the
/// body of the function that matches `AI.TargetFuncMatcher` in `AI.Code`. Given
/// the annotation line numbers and analysis outputs, `VerifyResults` checks
/// that the results from the analysis are correct.
///
/// Requirements:
///
///   `AnalysisT` contains a type `Lattice`.
///
///   `Code`, `TargetFuncMatcher` and `MakeAnalysis` must be provided in `AI`.
///
///   `VerifyResults` must be provided.
template <typename AnalysisT>
llvm::Error
checkDataflow(AnalysisInputs<AnalysisT> AI,
              std::function<void(const llvm::DenseMap<unsigned, std::string> &,
                                 const AnalysisOutputs &)>
                  VerifyResults) {
  return checkDataflow<AnalysisT>(
      std::move(AI), [&VerifyResults](const AnalysisOutputs &AO) {
        auto AnnotationLinesAndContent =
            buildLineToAnnotationMapping(AO.ASTCtx.getSourceManager(), AO.Code);
        VerifyResults(AnnotationLinesAndContent, AO);
      });
}

/// Runs dataflow specified from `AI.MakeAnalysis` and `AI.PostVisitCFG` on the
/// body of the function that matches `AI.TargetFuncMatcher` in `AI.Code`. Given
/// the state computed at each annotated statement and analysis outputs,
/// `VerifyResults` checks that the results from the analysis are correct.
///
/// Requirements:
///
///   `AnalysisT` contains a type `Lattice`.
///
///   `Code`, `TargetFuncMatcher` and `MakeAnalysis` must be provided in `AI`.
///
///   `VerifyResults` must be provided.
///
///   Any annotations appearing in `Code` must come after a statement.
///
///   There can be at most one annotation attached per statement.
///
///   Annotations must not be repeated.
template <typename AnalysisT>
llvm::Error
checkDataflow(AnalysisInputs<AnalysisT> AI,
              std::function<void(const llvm::StringMap<DataflowAnalysisState<
                                     typename AnalysisT::Lattice>> &,
                                 const AnalysisOutputs &)>
                  VerifyResults) {
  // Compute mapping from nodes of annotated statements to the content in the
  // annotation.
  llvm::DenseMap<const Stmt *, std::string> StmtToAnnotations;
  auto SetupTest = [&StmtToAnnotations,
                    PrevSetupTest = std::move(AI.SetupTest)](
                       AnalysisOutputs &AO) -> llvm::Error {
    auto MaybeStmtToAnnotations = buildStatementToAnnotationMapping(
        cast<FunctionDecl>(AO.InitEnv.getDeclCtx()), AO.Code);
    if (!MaybeStmtToAnnotations) {
      return MaybeStmtToAnnotations.takeError();
    }
    StmtToAnnotations = std::move(*MaybeStmtToAnnotations);
    return PrevSetupTest ? PrevSetupTest(AO) : llvm::Error::success();
  };

  using StateT = DataflowAnalysisState<typename AnalysisT::Lattice>;

  // Save the states computed for program points immediately following annotated
  // statements. The saved states are keyed by the content of the annotation.
  llvm::StringMap<StateT> AnnotationStates;
  auto PostVisitCFG =
      [&StmtToAnnotations, &AnnotationStates,
       PrevPostVisitCFG = std::move(AI.PostVisitCFG)](
          ASTContext &Ctx, const CFGElement &Elt,
          const TransferStateForDiagnostics<typename AnalysisT::Lattice>
              &State) {
        if (PrevPostVisitCFG) {
          PrevPostVisitCFG(Ctx, Elt, State);
        }
        // FIXME: Extend retrieval of state for non statement constructs.
        auto Stmt = Elt.getAs<CFGStmt>();
        if (!Stmt)
          return;
        auto It = StmtToAnnotations.find(Stmt->getStmt());
        if (It == StmtToAnnotations.end())
          return;
        auto [_, InsertSuccess] = AnnotationStates.insert(
            {It->second, StateT{State.Lattice, State.Env}});
        (void)_;
        (void)InsertSuccess;
        assert(InsertSuccess);
      };
  return checkDataflow<AnalysisT>(
      std::move(AI)
          .withSetupTest(std::move(SetupTest))
          .withPostVisitCFG(std::move(PostVisitCFG)),
      [&VerifyResults, &AnnotationStates](const AnalysisOutputs &AO) {
        VerifyResults(AnnotationStates, AO);
      });
}

/// Returns the `ValueDecl` for the given identifier.
///
/// Requirements:
///
///   `Name` must be unique in `ASTCtx`.
const ValueDecl *findValueDecl(ASTContext &ASTCtx, llvm::StringRef Name);

/// Creates and owns constraints which are boolean values.
class ConstraintContext {
public:
  // Creates an atomic boolean value.
  BoolValue *atom() {
    Vals.push_back(std::make_unique<AtomicBoolValue>());
    return Vals.back().get();
  }

  // Creates an instance of the Top boolean value.
  BoolValue *top() {
    Vals.push_back(std::make_unique<TopBoolValue>());
    return Vals.back().get();
  }

  // Creates a boolean conjunction value.
  BoolValue *conj(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    Vals.push_back(
        std::make_unique<ConjunctionValue>(*LeftSubVal, *RightSubVal));
    return Vals.back().get();
  }

  // Creates a boolean disjunction value.
  BoolValue *disj(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    Vals.push_back(
        std::make_unique<DisjunctionValue>(*LeftSubVal, *RightSubVal));
    return Vals.back().get();
  }

  // Creates a boolean negation value.
  BoolValue *neg(BoolValue *SubVal) {
    Vals.push_back(std::make_unique<NegationValue>(*SubVal));
    return Vals.back().get();
  }

  // Creates a boolean implication value.
  BoolValue *impl(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    Vals.push_back(
        std::make_unique<ImplicationValue>(*LeftSubVal, *RightSubVal));
    return Vals.back().get();
  }

  // Creates a boolean biconditional value.
  BoolValue *iff(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    Vals.push_back(
        std::make_unique<BiconditionalValue>(*LeftSubVal, *RightSubVal));
    return Vals.back().get();
  }

private:
  std::vector<std::unique_ptr<BoolValue>> Vals;
};

} // namespace test
} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_TESTING_SUPPORT_H_
