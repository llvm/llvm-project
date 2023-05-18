//===--- AvoidEndlCheck.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidEndlCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void AvoidEndlCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          unless(isExpansionInSystemHeader()),
          anyOf(cxxOperatorCallExpr(
                    hasOverloadedOperatorName("<<"),
                    hasRHS(declRefExpr(to(namedDecl(hasName("::std::endl"))))
                               .bind("expr"))),
                callExpr(argumentCountIs(1),
                         callee(functionDecl(hasName("::std::endl"))))
                    .bind("expr"))),
      this);
}

void AvoidEndlCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Expression = Result.Nodes.getNodeAs<Expr>("expr");
  assert(Expression);
  assert(isa<DeclRefExpr>(Expression) || isa<CallExpr>(Expression));

  // FIXME: It would be great if we could transform
  // 'std::cout << "Hi" << std::endl;' into
  // 'std::cout << "Hi\n"';

  if (llvm::isa<DeclRefExpr>(Expression)) {
    // Handle the more common streaming '... << std::endl' case
    const CharSourceRange TokenRange =
        CharSourceRange::getTokenRange(Expression->getSourceRange());
    const StringRef SourceText = Lexer::getSourceText(
        TokenRange, *Result.SourceManager, Result.Context->getLangOpts());

    auto Diag = diag(Expression->getBeginLoc(),
                     "do not use '%0' with streams; use '\\n' instead")
                << SourceText;

    Diag << FixItHint::CreateReplacement(TokenRange, "'\\n'");
  } else {
    // Handle the less common function call 'std::endl(...)' case
    const auto *CallExpression = llvm::cast<CallExpr>(Expression);
    assert(CallExpression->getNumArgs() == 1);

    const StringRef SourceText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(
            CallExpression->getCallee()->getSourceRange()),
        *Result.SourceManager, Result.Context->getLangOpts());

    const CharSourceRange ArgTokenRange = CharSourceRange::getTokenRange(
        CallExpression->getArg(0)->getSourceRange());
    const StringRef ArgSourceText = Lexer::getSourceText(
        ArgTokenRange, *Result.SourceManager, Result.Context->getLangOpts());

    const std::string ReplacementString =
        std::string(ArgSourceText) + " << '\\n'";

    diag(CallExpression->getBeginLoc(),
         "do not use '%0' with streams; use '\\n' instead")
        << SourceText
        << FixItHint::CreateReplacement(
               CharSourceRange::getTokenRange(CallExpression->getSourceRange()),
               ReplacementString);
  }
}

} // namespace clang::tidy::performance
