//===--- NullCheckAfterDereferenceCheck.cpp - clang-tidy-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NullCheckAfterDereferenceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/Models/NullPointerAnalysisModel.h"
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
using dataflow::NullCheckAfterDereferenceDiagnoser;
using dataflow::NullPointerAnalysisModel;

static constexpr llvm::StringLiteral FuncID("fun");

struct ExpandedResult {
  SourceLocation WarningLoc;
  std::optional<SourceLocation> DerefLoc;
};

using ExpandedResultType =
    std::pair<std::vector<ExpandedResult>, std::vector<ExpandedResult>>;

static std::optional<ExpandedResultType>
analyzeFunction(const FunctionDecl &FuncDecl) {
  using dataflow::AdornedCFG;
  using dataflow::DataflowAnalysisState;
  using llvm::Expected;

  ASTContext &ASTCtx = FuncDecl.getASTContext();

  if (FuncDecl.getBody() == nullptr) {
    return std::nullopt;
  }

  Expected<AdornedCFG> Context =
      AdornedCFG::build(FuncDecl, *FuncDecl.getBody(), ASTCtx);
  if (!Context)
    return std::nullopt;

  dataflow::DataflowAnalysisContext AnalysisContext(
      std::make_unique<dataflow::WatchedLiteralsSolver>());
  dataflow::Environment Env(AnalysisContext, FuncDecl);
  NullPointerAnalysisModel Analysis(ASTCtx);
  NullCheckAfterDereferenceDiagnoser Diagnoser;
  NullCheckAfterDereferenceDiagnoser::ResultType Diagnostics;

  using State = DataflowAnalysisState<NullPointerAnalysisModel::Lattice>;
  using DetailMaybeStates = std::vector<std::optional<State>>;

  auto DiagnoserImpl = [&ASTCtx, &Diagnoser,
                        &Diagnostics](const CFGElement &Elt,
                                      const State &S) mutable -> void {
    auto EltDiagnostics = Diagnoser.diagnose(ASTCtx, &Elt, S.Env);
    llvm::move(EltDiagnostics.first, std::back_inserter(Diagnostics.first));
    llvm::move(EltDiagnostics.second, std::back_inserter(Diagnostics.second)); 
  };

  Expected<DetailMaybeStates> BlockToOutputState =
      dataflow::runDataflowAnalysis(*Context, Analysis, Env, DiagnoserImpl);

  if (llvm::Error E = BlockToOutputState.takeError()) {
    llvm::dbgs() << "Dataflow analysis failed: " << llvm::toString(std::move(E))
                 << ".\n";
    return std::nullopt;
  }

  ExpandedResultType ExpandedDiagnostics;

  llvm::transform(Diagnostics.first,
                  std::back_inserter(ExpandedDiagnostics.first),
                  [&](SourceLocation WarningLoc) -> ExpandedResult {
                    if (auto Val = Diagnoser.WarningLocToVal[WarningLoc];
                        auto DerefExpr = Diagnoser.ValToDerefLoc[Val]) {
                      return {WarningLoc, DerefExpr->getBeginLoc()};
                    }

                    return {WarningLoc, std::nullopt};
                  });

  llvm::transform(Diagnostics.second,
                  std::back_inserter(ExpandedDiagnostics.second),
                  [&](SourceLocation WarningLoc) -> ExpandedResult {
                    if (auto Val = Diagnoser.WarningLocToVal[WarningLoc];
                        auto DerefExpr = Diagnoser.ValToDerefLoc[Val]) {
                      return {WarningLoc, DerefExpr->getBeginLoc()};
                    }

                    return {WarningLoc, std::nullopt};
                  });

  return ExpandedDiagnostics;
}

void NullCheckAfterDereferenceCheck::registerMatchers(MatchFinder *Finder) {
  using namespace ast_matchers;

  auto containsPointerValue =
      hasDescendant(NullPointerAnalysisModel::ptrValueMatcher());
  Finder->addMatcher(
      decl(anyOf(functionDecl(unless(isExpansionInSystemHeader()),
                              // FIXME: Remove the filter below when lambdas are
                              // well supported by the check.
                              unless(hasDeclContext(cxxRecordDecl(isLambda()))),
                              hasBody(containsPointerValue)),
                 cxxConstructorDecl(
                     unless(hasDeclContext(cxxRecordDecl(isLambda()))),
                     hasAnyConstructorInitializer(
                         withInitializer(containsPointerValue)))))
          .bind(FuncID),
      this);
}

void NullCheckAfterDereferenceCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (Result.SourceManager->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>(FuncID);
  assert(FuncDecl && "invalid FuncDecl matcher");
  if (FuncDecl->isTemplated())
    return;

  if (const auto Diagnostics = analyzeFunction(*FuncDecl)) {
    const auto &[CheckWhenNullLocations, CheckWhenNonnullLocations] =
        *Diagnostics;

    for (const auto [WarningLoc, DerefLoc] : CheckWhenNonnullLocations) {
      diag(WarningLoc, "pointer value is checked even though "
                       "it cannot be null at this point");

      if (DerefLoc) {
        diag(*DerefLoc,
             "one of the locations where the pointer's value cannot be null",
             DiagnosticIDs::Note);
      }
    }

    for (const auto [WarningLoc, DerefLoc] : CheckWhenNullLocations) {
      diag(WarningLoc,
           "pointer value is checked but it can only be null at this point");

      if (DerefLoc) {
        diag(*DerefLoc,
             "one of the locations where the pointer's value can only be null",
             DiagnosticIDs::Note);
      }
    }
  }
}

} // namespace clang::tidy::bugprone
