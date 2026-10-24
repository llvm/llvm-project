//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedExpectedAccessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedExpectedAccessModel.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace clang::tidy::bugprone {
using ast_matchers::MatchFinder;
using dataflow::UncheckedExpectedAccessDiagnoser;
using dataflow::UncheckedExpectedAccessDiagnostic;
using dataflow::UncheckedExpectedAccessModel;

static constexpr StringRef FuncID = "fun";

void UncheckedExpectedAccessCheck::registerMatchers(MatchFinder *Finder) {
  using namespace ast_matchers;

  const auto HasExpectedCallDescendant =
      hasDescendant(UncheckedExpectedAccessModel::callToExpectedClass());
  Finder->addMatcher(
      decl(anyOf(functionDecl(
                     // FIXME: Remove the filter below when lambdas are
                     // well supported by the check.
                     unless(hasDeclContext(cxxRecordDecl(isLambda()))),
                     hasBody(HasExpectedCallDescendant)),
                 cxxConstructorDecl(hasAnyConstructorInitializer(
                     withInitializer(HasExpectedCallDescendant)))))
          .bind(FuncID),
      this);
}

void UncheckedExpectedAccessCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (Result.SourceManager->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>(FuncID);
  if (FuncDecl->isTemplated())
    return;

  UncheckedExpectedAccessDiagnoser Diagnoser;
  if (llvm::Expected<SmallVector<UncheckedExpectedAccessDiagnostic>> Diags =
          dataflow::diagnoseFunction<UncheckedExpectedAccessModel,
                                     UncheckedExpectedAccessDiagnostic>(
              *FuncDecl, *Result.Context, Diagnoser))
    for (const UncheckedExpectedAccessDiagnostic &Diag : *Diags)
      diag(Diag.Loc, "unchecked access to '%0' %select{value|error}1")
          << Diag.TypeName << llvm::to_underlying(Diag.Kind);
  else
    llvm::consumeError(Diags.takeError());
}

} // namespace clang::tidy::bugprone
