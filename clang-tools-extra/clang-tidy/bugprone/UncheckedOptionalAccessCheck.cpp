//===--- UncheckedOptionalAccessCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedOptionalAccessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <vector>

namespace clang {
namespace tidy {
namespace bugprone {
using ast_matchers::MatchFinder;
using dataflow::SourceLocationsLattice;
using dataflow::UncheckedOptionalAccessDiagnoser;
using dataflow::UncheckedOptionalAccessModel;
using llvm::Optional;

static constexpr llvm::StringLiteral FuncID("fun");

static Optional<std::vector<SourceLocation>>
analyzeFunction(const FunctionDecl &FuncDecl, ASTContext &ASTCtx) {
  using dataflow::ControlFlowContext;
  using dataflow::DataflowAnalysisState;
  using llvm::Expected;

  Expected<ControlFlowContext> Context =
      ControlFlowContext::build(&FuncDecl, FuncDecl.getBody(), &ASTCtx);
  if (!Context)
    return llvm::None;

  dataflow::DataflowAnalysisContext AnalysisContext(
      std::make_unique<dataflow::WatchedLiteralsSolver>());
  dataflow::Environment Env(AnalysisContext, FuncDecl);
  UncheckedOptionalAccessModel Analysis(ASTCtx);
  UncheckedOptionalAccessDiagnoser Diagnoser;
  std::vector<SourceLocation> Diagnostics;
  Expected<std::vector<Optional<DataflowAnalysisState<SourceLocationsLattice>>>>
      BlockToOutputState = dataflow::runDataflowAnalysis(
          *Context, Analysis, Env,
          [&ASTCtx, &Diagnoser,
           &Diagnostics](const Stmt *Stmt,
                         const DataflowAnalysisState<SourceLocationsLattice>
                             &State) mutable {
            auto StmtDiagnostics = Diagnoser.diagnose(ASTCtx, Stmt, State.Env);
            llvm::move(StmtDiagnostics, std::back_inserter(Diagnostics));
          });
  if (!BlockToOutputState)
    return llvm::None;

  return Diagnostics;
}

void UncheckedOptionalAccessCheck::registerMatchers(MatchFinder *Finder) {
  using namespace ast_matchers;

  auto HasOptionalCallDescendant = hasDescendant(callExpr(callee(cxxMethodDecl(
      ofClass(UncheckedOptionalAccessModel::optionalClassDecl())))));
  Finder->addMatcher(
      decl(anyOf(functionDecl(unless(isExpansionInSystemHeader()),
                              // FIXME: Remove the filter below when lambdas are
                              // well supported by the check.
                              unless(hasDeclContext(cxxRecordDecl(isLambda()))),
                              hasBody(HasOptionalCallDescendant)),
                 cxxConstructorDecl(hasAnyConstructorInitializer(
                     withInitializer(HasOptionalCallDescendant)))))
          .bind(FuncID),
      this);
}

void UncheckedOptionalAccessCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (Result.SourceManager->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>(FuncID);
  if (FuncDecl->isTemplated())
    return;

  if (Optional<std::vector<SourceLocation>> Errors =
          analyzeFunction(*FuncDecl, *Result.Context))
    for (const SourceLocation &Loc : *Errors)
      diag(Loc, "unchecked access to optional value");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
