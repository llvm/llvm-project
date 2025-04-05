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
using Diagnoser = NullCheckAfterDereferenceDiagnoser;

static constexpr llvm::StringLiteral FuncID("fun");

struct ExpandedResult {
  Diagnoser::DiagnosticEntry Entry;
  std::optional<SourceLocation> DerefLoc;
};

using ExpandedResultType = llvm::SmallVector<ExpandedResult>;

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
  Diagnoser Diagnoser;

  Expected<Diagnoser::ResultType> Diagnostics =
      dataflow::diagnoseFunction<NullPointerAnalysisModel, Diagnoser::DiagnosticEntry>(
      FuncDecl, ASTCtx, Diagnoser);

  
  if (llvm::Error E = Diagnostics.takeError()) {
    llvm::dbgs() << "Dataflow analysis failed: " << llvm::toString(std::move(E))
                 << ".\n";
    return std::nullopt;
  }

  ExpandedResultType ExpandedDiagnostics;

  llvm::transform(*Diagnostics,
                  std::back_inserter(ExpandedDiagnostics),
                  [&](Diagnoser::DiagnosticEntry Entry) -> ExpandedResult {
                    if (auto Val = Diagnoser.WarningLocToVal[Entry.Location];
                        auto DerefExpr = Diagnoser.ValToDerefLoc[Val]) {
                      return {Entry, DerefExpr->getBeginLoc()};
                    }

                    return {Entry, std::nullopt};
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
    for (const auto [Entry, DerefLoc] : *Diagnostics) {
      const auto [WarningLoc, Type] = Entry;

      switch (Type) {
      case Diagnoser::DiagnosticType::CheckAfterDeref:
        diag(WarningLoc, "pointer value is checked even though "
                         "it cannot be null at this point");

        if (DerefLoc) {
          diag(*DerefLoc,
               "one of the locations where the pointer's value cannot be null",
               DiagnosticIDs::Note);
        }
        break;
      case Diagnoser::DiagnosticType::CheckWhenNull:
        diag(WarningLoc,
             "pointer value is checked but it can only be null at this point");

        if (DerefLoc) {
          diag(*DerefLoc,
               "one of the locations where the pointer's value can only be null",
               DiagnosticIDs::Note);
        }
        break;
      }
    }
  }
}

} // namespace clang::tidy::bugprone
