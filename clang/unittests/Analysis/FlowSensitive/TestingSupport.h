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
#include <optional>
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
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/LLVM.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Annotations/Annotations.h"

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
  llvm::ArrayRef<std::optional<TypeErasedDataflowAnalysisState>> BlockStates;
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
  AnalysisInputs<AnalysisT> &&
  withBuiltinOptions(DataflowAnalysisContext::Options Options) && {
    BuiltinOptions = std::move(Options);
    return std::move(*this);
  }
  AnalysisInputs<AnalysisT> &&
  withSolverFactory(std::function<std::unique_ptr<Solver>()> Factory) && {
    assert(Factory);
    SolverFactory = std::move(Factory);
    return std::move(*this);
  }

  /// Required. Input code that is analyzed.
  llvm::StringRef Code;
  /// Required. All functions that match this matcher are analyzed.
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
  /// Configuration options for the built-in model.
  DataflowAnalysisContext::Options BuiltinOptions;
  /// SAT solver factory.
  std::function<std::unique_ptr<Solver>()> SolverFactory = [] {
    return std::make_unique<WatchedLiteralsSolver>();
  };
};

/// Returns assertions based on annotations that are present after statements in
/// `AnnotatedCode`.
llvm::Expected<llvm::DenseMap<const Stmt *, std::string>>
buildStatementToAnnotationMapping(const FunctionDecl *Func,
                                  llvm::Annotations AnnotatedCode);

/// Returns line numbers and content of the annotations in `AnnotatedCode`
/// within the token range `BoundingRange`.
llvm::DenseMap<unsigned, std::string> buildLineToAnnotationMapping(
    const SourceManager &SM, const LangOptions &LangOpts,
    SourceRange BoundingRange, llvm::Annotations AnnotatedCode);

/// Runs dataflow specified from `AI.MakeAnalysis` and `AI.PostVisitCFG` on all
/// functions that match `AI.TargetFuncMatcher` in `AI.Code`.  Given the
/// analysis outputs, `VerifyResults` checks that the results from the analysis
/// are correct.
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

  std::function<void(const CFGElement &,
                     const TypeErasedDataflowAnalysisState &)>
      TypeErasedPostVisitCFG = nullptr;
  if (AI.PostVisitCFG) {
    TypeErasedPostVisitCFG = [&AI, &Context](
                                 const CFGElement &Element,
                                 const TypeErasedDataflowAnalysisState &State) {
      AI.PostVisitCFG(Context, Element,
                      TransferStateForDiagnostics<typename AnalysisT::Lattice>(
                          llvm::any_cast<const typename AnalysisT::Lattice &>(
                              State.Lattice.Value),
                          State.Env));
    };
  }

  SmallVector<ast_matchers::BoundNodes, 1> MatchResult = ast_matchers::match(
      ast_matchers::functionDecl(ast_matchers::hasBody(ast_matchers::stmt()),
                                 AI.TargetFuncMatcher)
          .bind("target"),
      Context);
  if (MatchResult.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "didn't find any matching target functions");
  for (const ast_matchers::BoundNodes &BN : MatchResult) {
    // Get the AST node of the target function.
    const FunctionDecl *Target = BN.getNodeAs<FunctionDecl>("target");
    if (Target == nullptr)
      return llvm::make_error<llvm::StringError>(
          llvm::errc::invalid_argument, "Could not find the target function.");

    // Build the control flow graph for the target function.
    auto MaybeCFCtx = ControlFlowContext::build(*Target);
    if (!MaybeCFCtx) return MaybeCFCtx.takeError();
    auto &CFCtx = *MaybeCFCtx;

    // Initialize states for running dataflow analysis.
    DataflowAnalysisContext DACtx(AI.SolverFactory(),
                                  {/*Opts=*/AI.BuiltinOptions});
    Environment InitEnv(DACtx, *Target);
    auto Analysis = AI.MakeAnalysis(Context, InitEnv);

    AnalysisOutputs AO{AnnotatedCode, Context, Target, CFCtx,
                       Analysis,      InitEnv, {}};

    // Additional test setup.
    if (AI.SetupTest) {
      if (auto Error = AI.SetupTest(AO)) return Error;
    }

    // If successful, the dataflow analysis returns a mapping from block IDs to
    // the post-analysis states for the CFG blocks that have been evaluated.
    llvm::Expected<std::vector<std::optional<TypeErasedDataflowAnalysisState>>>
        MaybeBlockStates = runTypeErasedDataflowAnalysis(
            CFCtx, Analysis, InitEnv, TypeErasedPostVisitCFG);
    if (!MaybeBlockStates) return MaybeBlockStates.takeError();
    AO.BlockStates = *MaybeBlockStates;

    // Verify dataflow analysis outputs.
    VerifyResults(AO);
  }

  return llvm::Error::success();
}

/// Runs dataflow specified from `AI.MakeAnalysis` and `AI.PostVisitCFG` on all
/// functions that match `AI.TargetFuncMatcher` in `AI.Code`. Given the
/// annotation line numbers and analysis outputs, `VerifyResults` checks that
/// the results from the analysis are correct.
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
        auto AnnotationLinesAndContent = buildLineToAnnotationMapping(
            AO.ASTCtx.getSourceManager(), AO.ASTCtx.getLangOpts(),
            AO.Target->getSourceRange(), AO.Code);
        VerifyResults(AnnotationLinesAndContent, AO);
      });
}

/// Runs dataflow specified from `AI.MakeAnalysis` and `AI.PostVisitCFG` on all
/// functions that match `AI.TargetFuncMatcher` in `AI.Code`. Given the state
/// computed at each annotated statement and analysis outputs, `VerifyResults`
/// checks that the results from the analysis are correct.
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
            {It->second, StateT{State.Lattice, State.Env.fork()}});
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

        // `checkDataflow()` can analyze more than one function.  Reset the
        // variables to prepare for analyzing the next function.
        AnnotationStates.clear();
      });
}

using BuiltinOptions = DataflowAnalysisContext::Options;

/// Runs dataflow on function named `TargetFun` in `Code` with a `NoopAnalysis`
/// and calls `VerifyResults` to verify the results.
llvm::Error checkDataflowWithNoopAnalysis(
    llvm::StringRef Code,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults = [](const auto &, auto &) {},
    DataflowAnalysisOptions Options = {BuiltinOptions()},
    LangStandard::Kind Std = LangStandard::lang_cxx17,
    llvm::StringRef TargetFun = "target");

/// Runs dataflow on function matched by `TargetFuncMatcher` in `Code` with a
/// `NoopAnalysis` and calls `VerifyResults` to verify the results.
llvm::Error checkDataflowWithNoopAnalysis(
    llvm::StringRef Code,
    ast_matchers::internal::Matcher<FunctionDecl> TargetFuncMatcher,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             ASTContext &)>
        VerifyResults = [](const auto &, auto &) {},
    DataflowAnalysisOptions Options = {BuiltinOptions()},
    LangStandard::Kind Std = LangStandard::lang_cxx17,
    std::function<llvm::StringMap<QualType>(QualType)> SyntheticFieldCallback =
        {});

/// Returns the `ValueDecl` for the given identifier.
///
/// Requirements:
///
///   `Name` must be unique in `ASTCtx`.
const ValueDecl *findValueDecl(ASTContext &ASTCtx, llvm::StringRef Name);

/// Returns the `IndirectFieldDecl` for the given identifier.
///
/// Requirements:
///
///   `Name` must be unique in `ASTCtx`.
const IndirectFieldDecl *findIndirectFieldDecl(ASTContext &ASTCtx,
                                               llvm::StringRef Name);

/// Returns the storage location (of type `LocT`) for the given identifier.
/// `LocT` must be a subclass of `StorageLocation` and must be of the
/// appropriate type.
///
/// Requirements:
///
///   `Name` must be unique in `ASTCtx`.
template <class LocT>
LocT &getLocForDecl(ASTContext &ASTCtx, const Environment &Env,
                    llvm::StringRef Name) {
  const ValueDecl *VD = findValueDecl(ASTCtx, Name);
  assert(VD != nullptr);
  return *cast<LocT>(Env.getStorageLocation(*VD));
}

/// Returns the value (of type `ValueT`) for the given identifier.
/// `ValueT` must be a subclass of `Value` and must be of the appropriate type.
///
/// Requirements:
///
///   `Name` must be unique in `ASTCtx`.
template <class ValueT>
ValueT &getValueForDecl(ASTContext &ASTCtx, const Environment &Env,
                        llvm::StringRef Name) {
  const ValueDecl *VD = findValueDecl(ASTCtx, Name);
  assert(VD != nullptr);
  return *cast<ValueT>(Env.getValue(*VD));
}

/// Returns the value of a `Field` on the record referenced by `Loc.`
/// Returns null if `Loc` is null.
inline Value *getFieldValue(const RecordStorageLocation *Loc,
                            const ValueDecl &Field, const Environment &Env) {
  if (Loc == nullptr)
    return nullptr;
  StorageLocation *FieldLoc = Loc->getChild(Field);
  if (FieldLoc == nullptr)
    return nullptr;
  return Env.getValue(*FieldLoc);
}

/// Creates and owns constraints which are boolean values.
class ConstraintContext {
  unsigned NextAtom = 0;
  llvm::BumpPtrAllocator A;

  const Formula *make(Formula::Kind K,
                      llvm::ArrayRef<const Formula *> Operands) {
    return &Formula::create(A, K, Operands);
  }

public:
  // Returns a reference to a fresh atomic variable.
  const Formula *atom() {
    return &Formula::create(A, Formula::AtomRef, {}, NextAtom++);
  }

  // Returns a reference to a literal boolean value.
  const Formula *literal(bool B) {
    return &Formula::create(A, Formula::Literal, {}, B);
  }

  // Creates a boolean conjunction.
  const Formula *conj(const Formula *LHS, const Formula *RHS) {
    return make(Formula::And, {LHS, RHS});
  }

  // Creates a boolean disjunction.
  const Formula *disj(const Formula *LHS, const Formula *RHS) {
    return make(Formula::Or, {LHS, RHS});
  }

  // Creates a boolean negation.
  const Formula *neg(const Formula *Operand) {
    return make(Formula::Not, {Operand});
  }

  // Creates a boolean implication.
  const Formula *impl(const Formula *LHS, const Formula *RHS) {
    return make(Formula::Implies, {LHS, RHS});
  }

  // Creates a boolean biconditional.
  const Formula *iff(const Formula *LHS, const Formula *RHS) {
    return make(Formula::Equal, {LHS, RHS});
  }
};

/// Parses a list of formulas, separated by newlines, and returns them.
/// On parse errors, calls `ADD_FAILURE()` to fail the current test.
std::vector<const Formula *> parseFormulas(Arena &A, StringRef Lines);

} // namespace test
} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_TESTING_SUPPORT_H_
