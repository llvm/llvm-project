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
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gtest/gtest.h"
#include <functional>
#include <memory>
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

// Creates a CFG from the body of the function that matches `func_matcher`,
// suitable to testing a dataflow analysis.
std::pair<const FunctionDecl *, std::unique_ptr<CFG>>
buildCFG(ASTContext &Context,
         ast_matchers::internal::Matcher<FunctionDecl> FuncMatcher);

// Runs dataflow on the body of the function that matches `func_matcher` in code
// snippet `code`. Requires: `Analysis` contains a type `Lattice`.
template <typename AnalysisT>
void checkDataflow(
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
  auto Unit = tooling::buildASTFromCodeWithArgs(AnnotatedCode.code(), Args);
  auto &Context = Unit->getASTContext();

  if (Context.getDiagnostics().getClient()->getNumErrors() != 0) {
    FAIL() << "Source file has syntax or type errors, they were printed to "
              "the test log";
  }

  std::pair<const FunctionDecl *, std::unique_ptr<CFG>> CFGResult =
      buildCFG(Context, FuncMatcher);
  const auto *F = CFGResult.first;
  auto Cfg = std::move(CFGResult.second);
  ASSERT_TRUE(F != nullptr) << "Could not find target function";
  ASSERT_TRUE(Cfg != nullptr) << "Could not build control flow graph.";

  Environment Env;
  auto Analysis = MakeAnalysis(Context, Env);

  llvm::Expected<llvm::DenseMap<const clang::Stmt *, std::string>>
      StmtToAnnotations = buildStatementToAnnotationMapping(F, AnnotatedCode);
  if (auto E = StmtToAnnotations.takeError()) {
    FAIL() << "Failed to build annotation map: "
           << llvm::toString(std::move(E));
    return;
  }
  auto &Annotations = *StmtToAnnotations;

  std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates =
      runTypeErasedDataflowAnalysis(*Cfg, Analysis, Env);

  if (BlockStates.empty()) {
    Expectations({}, Context);
    return;
  }

  // Compute a map from statement annotations to the state computed for
  // the program point immediately after the annotated statement.
  std::vector<std::pair<std::string, StateT>> Results;
  for (const CFGBlock *Block : *Cfg) {
    // Skip blocks that were not evaluated.
    if (!BlockStates[Block->getBlockID()].hasValue())
      continue;

    transferBlock(
        BlockStates, *Block, Env, Analysis,
        [&Results, &Annotations](const clang::CFGStmt &Stmt,
                                 const TypeErasedDataflowAnalysisState &State) {
          auto It = Annotations.find(Stmt.getStmt());
          if (It == Annotations.end())
            return;
          if (auto *Lattice = llvm::any_cast<typename AnalysisT::Lattice>(
                  &State.Lattice.Value)) {
            Results.emplace_back(It->second, StateT{*Lattice, State.Env});
          } else {
            FAIL() << "Could not cast lattice element to expected type.";
          }
        });
  }
  Expectations(Results, Context);
}

// Runs dataflow on the body of the function named `target_fun` in code snippet
// `code`.
template <typename AnalysisT>
void checkDataflow(
    llvm::StringRef Code, llvm::StringRef TargetFun,
    std::function<AnalysisT(ASTContext &, Environment &)> MakeAnalysis,
    std::function<void(
        llvm::ArrayRef<std::pair<
            std::string, DataflowAnalysisState<typename AnalysisT::Lattice>>>,
        ASTContext &)>
        Expectations,
    ArrayRef<std::string> Args,
    const tooling::FileContentMappings &VirtualMappedFiles = {}) {
  checkDataflow(Code, ast_matchers::hasName(TargetFun), std::move(MakeAnalysis),
                std::move(Expectations), Args, VirtualMappedFiles);
}

} // namespace test
} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_TESTING_SUPPORT_H_
