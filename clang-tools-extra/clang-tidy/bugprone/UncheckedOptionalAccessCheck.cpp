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
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <optional>
#include <vector>

namespace clang::tidy::bugprone {
using ast_matchers::MatchFinder;
using dataflow::UncheckedOptionalAccessDiagnoser;
using dataflow::UncheckedOptionalAccessModel;
using dataflow::UncheckedOptionalAccessModelOptions;

static constexpr llvm::StringLiteral FuncID("fun");

static std::optional<std::vector<SourceLocation>>
analyzeFunction(const FunctionDecl &FuncDecl, ASTContext &ASTCtx,
                UncheckedOptionalAccessModelOptions ModelOptions) {
  using dataflow::ControlFlowContext;
  using dataflow::DataflowAnalysisState;
  using llvm::Expected;

  Expected<ControlFlowContext> Context =
      ControlFlowContext::build(FuncDecl, *FuncDecl.getBody(), ASTCtx);
  if (!Context)
    return std::nullopt;

  dataflow::DataflowAnalysisContext AnalysisContext(
      std::make_unique<dataflow::WatchedLiteralsSolver>());
  dataflow::Environment Env(AnalysisContext, FuncDecl);
  UncheckedOptionalAccessModel Analysis(ASTCtx);
  UncheckedOptionalAccessDiagnoser Diagnoser(ModelOptions);
  std::vector<SourceLocation> Diagnostics;
  Expected<std::vector<std::optional<
      DataflowAnalysisState<UncheckedOptionalAccessModel::Lattice>>>>
      BlockToOutputState = dataflow::runDataflowAnalysis(
          *Context, Analysis, Env,
          [&ASTCtx, &Diagnoser, &Diagnostics](
              const CFGElement &Elt,
              const DataflowAnalysisState<UncheckedOptionalAccessModel::Lattice>
                  &State) mutable {
            auto EltDiagnostics = Diagnoser.diagnose(ASTCtx, &Elt, State.Env);
            llvm::move(EltDiagnostics, std::back_inserter(Diagnostics));
          });
  if (!BlockToOutputState)
    return std::nullopt;

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

  if (std::optional<std::vector<SourceLocation>> Errors =
          analyzeFunction(*FuncDecl, *Result.Context, ModelOptions))
    for (const SourceLocation &Loc : *Errors)
      diag(Loc, "unchecked access to optional value");
}

} // namespace clang::tidy::bugprone
