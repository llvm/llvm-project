//===--- UncheckedOptionalAccessCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedOptionalAccessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace clang::tidy::bugprone {
using ast_matchers::MatchFinder;
using dataflow::UncheckedOptionalAccessDiagnoser;
using dataflow::UncheckedOptionalAccessModel;

static constexpr llvm::StringLiteral FuncID("fun");

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

void UncheckedOptionalAccessCheck::onStartOfTranslationUnit() {
  // Reset the flag for each TU.
  is_test_tu_ = false;
}

void UncheckedOptionalAccessCheck::check(
    const MatchFinder::MatchResult &Result) {
  // The googletest assertion macros are not currently recognized, so we have
  // many false positives in tests. So, do not check functions in a test TU.
  if (is_test_tu_ ||
      Result.SourceManager->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  // Look for two (public) googletest macros; if found, we'll mark this TU as a
  // test TU. We look for ASSERT_TRUE because it is a problematic macro that
  // we don't (yet) support, and GTEST_TEST to disambiguate ASSERT_TRUE.
  if (Result.Context->Idents.get("ASSERT_TRUE").hasMacroDefinition() &&
      Result.Context->Idents.get("GTEST_TEST").hasMacroDefinition()) {
    is_test_tu_ = true;
    return;
  }

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>(FuncID);
  if (FuncDecl->isTemplated())
    return;

  UncheckedOptionalAccessDiagnoser Diagnoser(ModelOptions);
  // FIXME: Allow user to set the (defaulted) SAT iterations max for
  // `diagnoseFunction` with config options.
  if (llvm::Expected<llvm::SmallVector<SourceLocation>> Locs =
          dataflow::diagnoseFunction<UncheckedOptionalAccessModel,
                                     SourceLocation>(*FuncDecl, *Result.Context,
                                                     Diagnoser))
    for (const SourceLocation &Loc : *Locs)
      diag(Loc, "unchecked access to optional value");
  else
    llvm::consumeError(Locs.takeError());
}

} // namespace clang::tidy::bugprone
