//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DurationFactoryFloatCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::abseil {

// Returns `true` if `Range` is inside a macro definition.
static bool insideMacroDefinition(const MatchFinder::MatchResult &Result,
                                  SourceRange Range) {
  return !clang::Lexer::makeFileCharRange(
              clang::CharSourceRange::getCharRange(Range),
              *Result.SourceManager, Result.Context->getLangOpts())
              .isValid();
}

void DurationFactoryFloatCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(DurationFactoryFunction())),
               hasArgument(0, anyOf(cxxStaticCastExpr(hasDestinationType(
                                        realFloatingPointType())),
                                    cStyleCastExpr(hasDestinationType(
                                        realFloatingPointType())),
                                    cxxFunctionalCastExpr(hasDestinationType(
                                        realFloatingPointType())),
                                    floatLiteral())))
          .bind("call"),
      this);
}

void DurationFactoryFloatCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>("call");

  // Don't try and replace things inside of macro definitions.
  if (insideMacroDefinition(Result, MatchedCall->getSourceRange()))
    return;

  const Expr *Arg = MatchedCall->getArg(0)->IgnoreImpCasts();
  // Arguments which are macros are ignored.
  if (Arg->getBeginLoc().isMacroID())
    return;

  std::optional<std::string> SimpleArg = stripFloatCast(Result, *Arg);
  if (!SimpleArg)
    SimpleArg = stripFloatLiteralFraction(Result, *Arg);

  if (SimpleArg) {
    diag(MatchedCall->getBeginLoc(), "use the integer version of absl::%0")
        << MatchedCall->getDirectCallee()->getName()
        << FixItHint::CreateReplacement(Arg->getSourceRange(), *SimpleArg);
  }
}

} // namespace clang::tidy::abseil
