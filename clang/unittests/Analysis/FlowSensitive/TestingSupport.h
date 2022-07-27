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
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/LLVM.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
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

// Returns assertions based on annotations that are present after statements in
// `AnnotatedCode`.
llvm::Expected<llvm::DenseMap<const Stmt *, std::string>>
buildStatementToAnnotationMapping(const FunctionDecl *Func,
                                  llvm::Annotations AnnotatedCode);

struct AnalysisData {
  ASTContext &ASTCtx;
  const ControlFlowContext &CFCtx;
  const Environment &Env;
  TypeErasedDataflowAnalysis &Analysis;
  llvm::DenseMap<const clang::Stmt *, std::string> &Annotations;
  std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> &BlockStates;
};

template <typename AnalysisT>
llvm::Error checkDataflow(
    llvm::StringRef Code,
    ast_matchers::internal::Matcher<FunctionDecl> TargetFuncMatcher,
    std::function<AnalysisT(ASTContext &, Environment &)> MakeAnalysis,
    std::function<void(ASTContext &, const Stmt *,
                       const TypeErasedDataflowAnalysisState &)>
        PostVisitStmt,
    std::function<void(AnalysisData)> VerifyResults, ArrayRef<std::string> Args,
    const tooling::FileContentMappings &VirtualMappedFiles = {}) {
  llvm::Annotations AnnotatedCode(Code);
  auto Unit = tooling::buildASTFromCodeWithArgs(
      AnnotatedCode.code(), Args, "input.cc", "clang-dataflow-test",
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(), VirtualMappedFiles);
  auto &Context = Unit->getASTContext();

  if (Context.getDiagnostics().getClient()->getNumErrors() != 0) {
    return llvm::make_error<llvm::StringError>(
        llvm::errc::invalid_argument, "Source file has syntax or type errors, "
                                      "they were printed to the test log");
  }

  const FunctionDecl *F = ast_matchers::selectFirst<FunctionDecl>(
      "target",
      ast_matchers::match(ast_matchers::functionDecl(
                              ast_matchers::isDefinition(), TargetFuncMatcher)
                              .bind("target"),
                          Context));
  if (F == nullptr)
    return llvm::make_error<llvm::StringError>(
        llvm::errc::invalid_argument, "Could not find target function.");

  auto CFCtx = ControlFlowContext::build(F, F->getBody(), &F->getASTContext());
  if (!CFCtx)
    return CFCtx.takeError();

  DataflowAnalysisContext DACtx(std::make_unique<WatchedLiteralsSolver>());
  Environment Env(DACtx, *F);
  auto Analysis = MakeAnalysis(Context, Env);

  std::function<void(const Stmt *, const TypeErasedDataflowAnalysisState &)>
      PostVisitStmtClosure = nullptr;
  if (PostVisitStmt != nullptr) {
    PostVisitStmtClosure = [&PostVisitStmt, &Context](
                               const Stmt *Stmt,
                               const TypeErasedDataflowAnalysisState &State) {
      PostVisitStmt(Context, Stmt, State);
    };
  }

  llvm::Expected<llvm::DenseMap<const clang::Stmt *, std::string>>
      StmtToAnnotations = buildStatementToAnnotationMapping(F, AnnotatedCode);
  if (!StmtToAnnotations)
    return StmtToAnnotations.takeError();
  auto &Annotations = *StmtToAnnotations;

  llvm::Expected<std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>>
      MaybeBlockStates = runTypeErasedDataflowAnalysis(*CFCtx, Analysis, Env,
                                                       PostVisitStmtClosure);
  if (!MaybeBlockStates)
    return MaybeBlockStates.takeError();
  auto &BlockStates = *MaybeBlockStates;

  AnalysisData AnalysisData{Context,  *CFCtx,      Env,
                            Analysis, Annotations, BlockStates};
  VerifyResults(AnalysisData);
  return llvm::Error::success();
}

// Runs dataflow on the body of the function that matches `TargetFuncMatcher` in
// code snippet `Code`. Requires: `AnalysisT` contains a type `Lattice`.
template <typename AnalysisT>
llvm::Error checkDataflow(
    llvm::StringRef Code,
    ast_matchers::internal::Matcher<FunctionDecl> TargetFuncMatcher,
    std::function<AnalysisT(ASTContext &, Environment &)> MakeAnalysis,
    std::function<void(
        llvm::ArrayRef<std::pair<
            std::string, DataflowAnalysisState<typename AnalysisT::Lattice>>>,
        ASTContext &)>
        VerifyResults,
    ArrayRef<std::string> Args,
    const tooling::FileContentMappings &VirtualMappedFiles = {}) {
  using StateT = DataflowAnalysisState<typename AnalysisT::Lattice>;

  return checkDataflow(
      Code, std::move(TargetFuncMatcher), std::move(MakeAnalysis),
      /*PostVisitStmt=*/nullptr,
      [&VerifyResults](AnalysisData AnalysisData) {
        if (AnalysisData.BlockStates.empty()) {
          VerifyResults({}, AnalysisData.ASTCtx);
          return;
        }

        auto &Annotations = AnalysisData.Annotations;

        // Compute a map from statement annotations to the state computed for
        // the program point immediately after the annotated statement.
        std::vector<std::pair<std::string, StateT>> Results;
        for (const CFGBlock *Block : AnalysisData.CFCtx.getCFG()) {
          // Skip blocks that were not evaluated.
          if (!AnalysisData.BlockStates[Block->getBlockID()])
            continue;

          transferBlock(
              AnalysisData.CFCtx, AnalysisData.BlockStates, *Block,
              AnalysisData.Env, AnalysisData.Analysis,
              [&Results,
               &Annotations](const clang::CFGStmt &Stmt,
                             const TypeErasedDataflowAnalysisState &State) {
                auto It = Annotations.find(Stmt.getStmt());
                if (It == Annotations.end())
                  return;
                auto *Lattice = llvm::any_cast<typename AnalysisT::Lattice>(
                    &State.Lattice.Value);
                Results.emplace_back(It->second, StateT{*Lattice, State.Env});
              });
        }
        VerifyResults(Results, AnalysisData.ASTCtx);
      },
      Args, VirtualMappedFiles);
}

// Runs dataflow on the body of the function named `target_fun` in code snippet
// `code`.
template <typename AnalysisT>
llvm::Error checkDataflow(
    llvm::StringRef Code, llvm::StringRef TargetFun,
    std::function<AnalysisT(ASTContext &, Environment &)> MakeAnalysis,
    std::function<void(
        llvm::ArrayRef<std::pair<
            std::string, DataflowAnalysisState<typename AnalysisT::Lattice>>>,
        ASTContext &)>
        VerifyResults,
    ArrayRef<std::string> Args,
    const tooling::FileContentMappings &VirtualMappedFiles = {}) {
  return checkDataflow(Code, ast_matchers::hasName(TargetFun),
                       std::move(MakeAnalysis), std::move(VerifyResults), Args,
                       VirtualMappedFiles);
}

/// Returns the `ValueDecl` for the given identifier.
///
/// Requirements:
///
///  `Name` must be unique in `ASTCtx`.
const ValueDecl *findValueDecl(ASTContext &ASTCtx, llvm::StringRef Name);

/// Creates and owns constraints which are boolean values.
class ConstraintContext {
public:
  // Creates an atomic boolean value.
  BoolValue *atom() {
    Vals.push_back(std::make_unique<AtomicBoolValue>());
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
