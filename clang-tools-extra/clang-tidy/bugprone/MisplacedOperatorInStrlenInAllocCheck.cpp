//===--- MisplacedOperatorInStrlenInAllocCheck.cpp - clang-tidy------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MisplacedOperatorInStrlenInAllocCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void MisplacedOperatorInStrlenInAllocCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto StrLenFunc = functionDecl(anyOf(
      hasName("::strlen"), hasName("::std::strlen"), hasName("::strnlen"),
      hasName("::std::strnlen"), hasName("::strnlen_s"),
      hasName("::std::strnlen_s"), hasName("::wcslen"),
      hasName("::std::wcslen"), hasName("::wcsnlen"), hasName("::std::wcsnlen"),
      hasName("::wcsnlen_s"), hasName("std::wcsnlen_s")));

  const auto BadUse =
      callExpr(callee(StrLenFunc),
               hasAnyArgument(ignoringImpCasts(
                   binaryOperator(
                       hasOperatorName("+"),
                       hasRHS(ignoringParenImpCasts(integerLiteral(equals(1)))))
                       .bind("BinOp"))))
          .bind("StrLen");

  const auto BadArg = anyOf(
      allOf(unless(binaryOperator(
                hasOperatorName("+"), hasLHS(BadUse),
                hasRHS(ignoringParenImpCasts(integerLiteral(equals(1)))))),
            hasDescendant(BadUse)),
      BadUse);

  const auto Alloc0Func =
      functionDecl(anyOf(hasName("::malloc"), hasName("std::malloc"),
                         hasName("::alloca"), hasName("std::alloca")));
  const auto Alloc1Func =
      functionDecl(anyOf(hasName("::calloc"), hasName("std::calloc"),
                         hasName("::realloc"), hasName("std::realloc")));

  const auto Alloc0FuncPtr =
      varDecl(hasType(isConstQualified()),
              hasInitializer(ignoringParenImpCasts(
                  declRefExpr(hasDeclaration(Alloc0Func)))));
  const auto Alloc1FuncPtr =
      varDecl(hasType(isConstQualified()),
              hasInitializer(ignoringParenImpCasts(
                  declRefExpr(hasDeclaration(Alloc1Func)))));

  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
               callExpr(callee(decl(anyOf(Alloc0Func, Alloc0FuncPtr))),
                        hasArgument(0, BadArg))
                   .bind("Alloc")),
      this);
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
               callExpr(callee(decl(anyOf(Alloc1Func, Alloc1FuncPtr))),
                        hasArgument(1, BadArg))
                   .bind("Alloc")),
      this);
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
               cxxNewExpr(isArray(), hasArraySize(BadArg)).bind("Alloc")),
      this);
}

void MisplacedOperatorInStrlenInAllocCheck::check(
    const MatchFinder::MatchResult &Result) {
  const Expr *Alloc = Result.Nodes.getNodeAs<CallExpr>("Alloc");
  if (!Alloc)
    Alloc = Result.Nodes.getNodeAs<CXXNewExpr>("Alloc");
  assert(Alloc && "Matched node bound by 'Alloc' should be either 'CallExpr'"
         " or 'CXXNewExpr'");

  const auto *StrLen = Result.Nodes.getNodeAs<CallExpr>("StrLen");
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("BinOp");

  const StringRef StrLenText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(StrLen->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  const StringRef Arg0Text = Lexer::getSourceText(
      CharSourceRange::getTokenRange(StrLen->getArg(0)->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  const StringRef StrLenBegin = StrLenText.substr(0, StrLenText.find(Arg0Text));
  const StringRef StrLenEnd = StrLenText.substr(
      StrLenText.find(Arg0Text) + Arg0Text.size(), StrLenText.size());

  const StringRef LHSText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(BinOp->getLHS()->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  const StringRef RHSText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(BinOp->getRHS()->getSourceRange()),
      *Result.SourceManager, getLangOpts());

  auto Hint = FixItHint::CreateReplacement(
      StrLen->getSourceRange(),
      (StrLenBegin + LHSText + StrLenEnd + " + " + RHSText).str());

  diag(Alloc->getBeginLoc(),
       "addition operator is applied to the argument of %0 instead of its "
       "result")
      << StrLen->getDirectCallee()->getName() << Hint;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
