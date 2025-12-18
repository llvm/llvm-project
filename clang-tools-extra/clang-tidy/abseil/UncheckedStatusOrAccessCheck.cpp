//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedStatusOrAccessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedStatusOrAccessModel.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace clang::tidy::abseil {
using ast_matchers::MatchFinder;
using dataflow::statusor_model::UncheckedStatusOrAccessDiagnoser;
using dataflow::statusor_model::UncheckedStatusOrAccessModel;

static constexpr StringRef FuncID = "fun";

void UncheckedStatusOrAccessCheck::registerMatchers(MatchFinder *Finder) {
  using namespace ast_matchers;

  auto HasStatusOrCallDescendant =
      hasDescendant(callExpr(callee(cxxMethodDecl(ofClass(hasAnyName(
          "absl::StatusOr", "absl::internal_statusor::OperatorBase"))))));
  Finder->addMatcher(functionDecl(unless(isExpansionInSystemHeader()),
                                  hasBody(HasStatusOrCallDescendant))
                         .bind(FuncID),
                     this);
  Finder->addMatcher(
      cxxConstructorDecl(hasAnyConstructorInitializer(
                             withInitializer(HasStatusOrCallDescendant)))
          .bind(FuncID),
      this);
}

void UncheckedStatusOrAccessCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (Result.SourceManager->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>(FuncID);
  if (FuncDecl->isTemplated())
    return;

  UncheckedStatusOrAccessDiagnoser Diagnoser;
  if (llvm::Expected<llvm::SmallVector<SourceLocation>> Locs =
          dataflow::diagnoseFunction<UncheckedStatusOrAccessModel,
                                     SourceLocation>(*FuncDecl, *Result.Context,
                                                     Diagnoser))
    for (const SourceLocation &Loc : *Locs)
      diag(Loc, "unchecked access to 'absl::StatusOr' value");
  else
    llvm::consumeError(Locs.takeError());
}

bool UncheckedStatusOrAccessCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

} // namespace clang::tidy::abseil
