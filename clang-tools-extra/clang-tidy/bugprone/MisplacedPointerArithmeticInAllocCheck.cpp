//===--- MisplacedPointerArithmeticInAllocCheck.cpp - clang-tidy-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MisplacedPointerArithmeticInAllocCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void MisplacedPointerArithmeticInAllocCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto AllocFunc =
      functionDecl(hasAnyName("::malloc", "std::malloc", "::alloca", "::calloc",
                              "std::calloc", "::realloc", "std::realloc"));

  const auto AllocFuncPtr =
      varDecl(hasType(isConstQualified()),
              hasInitializer(ignoringParenImpCasts(
                  declRefExpr(hasDeclaration(AllocFunc)))));

  const auto AdditiveOperator = binaryOperator(hasAnyOperatorName("+", "-"));

  const auto IntExpr = expr(hasType(isInteger()));

  const auto AllocCall = callExpr(callee(decl(anyOf(AllocFunc, AllocFuncPtr))));

  Finder->addMatcher(
      binaryOperator(
          AdditiveOperator,
          hasLHS(anyOf(AllocCall, castExpr(hasSourceExpression(AllocCall)))),
          hasRHS(IntExpr))
          .bind("PtrArith"),
      this);

  const auto New = cxxNewExpr(unless(isArray()));

  Finder->addMatcher(binaryOperator(AdditiveOperator,
                                    hasLHS(anyOf(New, castExpr(New))),
                                    hasRHS(IntExpr))
                         .bind("PtrArith"),
                     this);

  const auto ArrayNew = cxxNewExpr(isArray());

  Finder->addMatcher(binaryOperator(AdditiveOperator,
                                    hasLHS(anyOf(ArrayNew, castExpr(ArrayNew))),
                                    hasRHS(IntExpr))
                         .bind("PtrArith"),
                     this);
}

void MisplacedPointerArithmeticInAllocCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *PtrArith = Result.Nodes.getNodeAs<BinaryOperator>("PtrArith");
  const Expr *AllocExpr = PtrArith->getLHS()->IgnoreParenCasts();
  std::string CallName;

  if (const auto *Call = dyn_cast<CallExpr>(AllocExpr)) {
    const NamedDecl *Func = Call->getDirectCallee();
    if (!Func) {
      Func = cast<NamedDecl>(Call->getCalleeDecl());
    }
    CallName = Func->getName().str();
  } else {
    const auto *New = cast<CXXNewExpr>(AllocExpr);
    if (New->isArray()) {
      CallName = "operator new[]";
    } else {
      const auto *CtrE = New->getConstructExpr();
      if (!CtrE || !CtrE->getArg(CtrE->getNumArgs() - 1)
                                     ->getType()
                                     ->isIntegralOrEnumerationType())
        return;
      CallName = "operator new";
    }
  }

  const SourceRange OldRParen = SourceRange(PtrArith->getLHS()->getEndLoc());
  const StringRef RParen =
      Lexer::getSourceText(CharSourceRange::getTokenRange(OldRParen),
                           *Result.SourceManager, getLangOpts());
  const SourceLocation NewRParen = Lexer::getLocForEndOfToken(
      PtrArith->getEndLoc(), 0, *Result.SourceManager, getLangOpts());

  diag(PtrArith->getBeginLoc(),
       "arithmetic operation is applied to the result of %0() instead of its "
       "size-like argument")
      << CallName << FixItHint::CreateRemoval(OldRParen)
      << FixItHint::CreateInsertion(NewRParen, RParen);
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
