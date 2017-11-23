//===--- MisplacedOperatorInStrlenInAllocCheck.cpp - clang-tidy------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MisplacedOperatorInStrlenInAllocCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

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
                   binaryOperator(allOf(hasOperatorName("+"),
                                        hasRHS(ignoringParenImpCasts(
                                            integerLiteral(equals(1))))))
                       .bind("BinOp"))))
          .bind("StrLen");

  const auto BadArg = anyOf(
      allOf(hasDescendant(BadUse),
            unless(binaryOperator(allOf(
                hasOperatorName("+"), hasLHS(BadUse),
                hasRHS(ignoringParenImpCasts(integerLiteral(equals(1)))))))),
      BadUse);

  const auto Alloc0Func =
      functionDecl(anyOf(hasName("::malloc"), hasName("std::malloc"),
                         hasName("::alloca"), hasName("std::alloca")));
  const auto Alloc1Func =
      functionDecl(anyOf(hasName("::calloc"), hasName("std::calloc"),
                         hasName("::realloc"), hasName("std::realloc")));

  Finder->addMatcher(
      callExpr(callee(Alloc0Func), hasArgument(0, BadArg)).bind("Alloc"), this);
  Finder->addMatcher(
      callExpr(callee(Alloc1Func), hasArgument(1, BadArg)).bind("Alloc"), this);
}

void MisplacedOperatorInStrlenInAllocCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Alloc = Result.Nodes.getNodeAs<CallExpr>("Alloc");
  const auto *StrLen = Result.Nodes.getNodeAs<CallExpr>("StrLen");
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("BinOp");

  const StringRef StrLenText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(StrLen->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  const StringRef StrLenBegin = StrLenText.substr(0, StrLenText.find('(') + 1);
  const StringRef Arg0Text = Lexer::getSourceText(
      CharSourceRange::getTokenRange(StrLen->getArg(0)->getSourceRange()),
      *Result.SourceManager, getLangOpts());
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

  diag(Alloc->getLocStart(),
       "addition operator is applied to the argument of %0 instead of its "
       "result") << StrLen->getDirectCallee()->getName() << Hint;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
