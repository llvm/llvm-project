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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Testing/Support/Annotations.h"
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

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

// Runs dataflow on the body of the function that matches `func_matcher` in code
// snippet `code`. Requires: `Analysis` contains a type `Lattice`.
template <typename AnalysisT>
llvm::Error checkDataflow(
    llvm::StringRef Code,
    ast_matchers::internal::Matcher<FunctionDecl> FuncMatcher,
    std::function<AnalysisT(ASTContext &, Environment &)> MakeAnalysis,
    std::function<void(
        llvm::ArrayRef<std::pair<
            std::string, DataflowAnalysisState<typename AnalysisT::Lattice>>>,
        ASTContext &)>
        Expectations,
    ArrayRef<std::string> Args,
    const tooling::FileContentMappings &VirtualMappedFiles = {}) {
  using StateT = DataflowAnalysisState<typename AnalysisT::Lattice>;

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
      ast_matchers::match(
          ast_matchers::functionDecl(ast_matchers::isDefinition(), FuncMatcher)
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

  llvm::Expected<llvm::DenseMap<const clang::Stmt *, std::string>>
      StmtToAnnotations = buildStatementToAnnotationMapping(F, AnnotatedCode);
  if (!StmtToAnnotations)
    return StmtToAnnotations.takeError();
  auto &Annotations = *StmtToAnnotations;

  llvm::Expected<std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>>
      MaybeBlockStates = runTypeErasedDataflowAnalysis(*CFCtx, Analysis, Env);
  if (!MaybeBlockStates)
    return MaybeBlockStates.takeError();
  auto &BlockStates = *MaybeBlockStates;

  if (BlockStates.empty()) {
    Expectations({}, Context);
    return llvm::Error::success();
  }

  // Compute a map from statement annotations to the state computed for
  // the program point immediately after the annotated statement.
  std::vector<std::pair<std::string, StateT>> Results;
  for (const CFGBlock *Block : CFCtx->getCFG()) {
    // Skip blocks that were not evaluated.
    if (!BlockStates[Block->getBlockID()].hasValue())
      continue;

    transferBlock(
        *CFCtx, BlockStates, *Block, Env, Analysis,
        [&Results, &Annotations](const clang::CFGStmt &Stmt,
                                 const TypeErasedDataflowAnalysisState &State) {
          auto It = Annotations.find(Stmt.getStmt());
          if (It == Annotations.end())
            return;
          auto *Lattice =
              llvm::any_cast<typename AnalysisT::Lattice>(&State.Lattice.Value);
          Results.emplace_back(It->second, StateT{*Lattice, State.Env});
        });
  }
  Expectations(Results, Context);
  return llvm::Error::success();
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
        Expectations,
    ArrayRef<std::string> Args,
    const tooling::FileContentMappings &VirtualMappedFiles = {}) {
  return checkDataflow(Code, ast_matchers::hasName(TargetFun),
                       std::move(MakeAnalysis), std::move(Expectations), Args,
                       VirtualMappedFiles);
}

/// Returns the `ValueDecl` for the given identifier.
///
/// Requirements:
///
///  `Name` must be unique in `ASTCtx`.
const ValueDecl *findValueDecl(ASTContext &ASTCtx, llvm::StringRef Name);

} // namespace test
} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_TESTING_SUPPORT_H_
