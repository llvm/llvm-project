//===--- DataflowDeadCodeCheck.cpp - clang-tidy-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DataflowDeadCodeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/Models/DeadCodeModel.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <clang/Analysis/FlowSensitive/AdornedCFG.h>
#include <memory>
#include <vector>

namespace clang::tidy::bugprone {

using ast_matchers::MatchFinder;
using dataflow::DeadCodeDiagnoser;
using dataflow::DeadCodeModel;
using dataflow::NoopAnalysis;
using Diagnoser = DeadCodeDiagnoser;

static constexpr llvm::StringLiteral FuncID("fun");

struct ExpandedResult {
  Diagnoser::DiagnosticEntry Entry;
  std::optional<SourceLocation> DerefLoc;
};

using ResultType = Diagnoser::ResultType;

static std::optional<ResultType> analyzeFunction(const FunctionDecl &FuncDecl) {
  using dataflow::AdornedCFG;
  using dataflow::DataflowAnalysisState;
  using llvm::Expected;

  ASTContext &ASTCtx = FuncDecl.getASTContext();

  if (!FuncDecl.doesThisDeclarationHaveABody()) {
    return std::nullopt;
  }

  Expected<AdornedCFG> Context =
      AdornedCFG::build(FuncDecl, *FuncDecl.getBody(), ASTCtx);
  if (!Context)
    return std::nullopt;

  dataflow::DataflowAnalysisContext AnalysisContext(
      std::make_unique<dataflow::WatchedLiteralsSolver>());
  dataflow::Environment Env(AnalysisContext, FuncDecl);
  DeadCodeModel Analysis(ASTCtx);
  Diagnoser Diagnoser;

  ResultType Diagnostics;

  if (llvm::Error E =
          dataflow::diagnoseFunction<DeadCodeModel, Diagnoser::DiagnosticEntry>(
              FuncDecl, ASTCtx, Diagnoser)
              .moveInto(Diagnostics)) {
    llvm::dbgs() << "Dataflow analysis failed: " << llvm::toString(std::move(E))
                 << ".\n";
    return std::nullopt;
  }

  return Diagnostics;
}

void DataflowDeadCodeCheck::registerMatchers(MatchFinder *Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      decl(
          anyOf(functionDecl(unless(isExpansionInSystemHeader()),
                             // FIXME: Remove the filter below when lambdas are
                             // well supported by the check.
                             unless(hasDeclContext(cxxRecordDecl(isLambda())))),
                cxxConstructorDecl(
                    unless(hasDeclContext(cxxRecordDecl(isLambda()))))))
          .bind(FuncID),
      this);
}

void DataflowDeadCodeCheck::check(const MatchFinder::MatchResult &Result) {
  if (Result.SourceManager->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>(FuncID);
  assert(FuncDecl && "invalid FuncDecl matcher");
  if (FuncDecl->isTemplated())
    return;

  if (const auto Diagnostics = analyzeFunction(*FuncDecl)) {
    for (const auto [Loc, Type] : *Diagnostics) {

      switch (Type) {
      case Diagnoser::DiagnosticType::AlwaysTrue:
        diag(Loc, "dead code - branching condition is always true");
        break;

      case Diagnoser::DiagnosticType::AlwaysFalse:
        diag(Loc, "dead code - branching condition is always false");
        break;
      }
    }
  }
}

} // namespace clang::tidy::bugprone
