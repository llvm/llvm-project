//===--- DurationFactoryFloatCheck.cpp - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DurationFactoryFloatCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

// Returns `true` if `Range` is inside a macro definition.
static bool InsideMacroDefinition(const MatchFinder::MatchResult &Result,
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
  if (InsideMacroDefinition(Result, MatchedCall->getSourceRange()))
    return;

  const Expr *Arg = MatchedCall->getArg(0)->IgnoreImpCasts();
  // Arguments which are macros are ignored.
  if (Arg->getBeginLoc().isMacroID())
    return;

  llvm::Optional<std::string> SimpleArg = stripFloatCast(Result, *Arg);
  if (!SimpleArg)
    SimpleArg = stripFloatLiteralFraction(Result, *Arg);

  if (SimpleArg) {
    diag(MatchedCall->getBeginLoc(),
         (llvm::Twine("use the integer version of absl::") +
          MatchedCall->getDirectCallee()->getName())
             .str())
        << FixItHint::CreateReplacement(Arg->getSourceRange(), *SimpleArg);
  }
}

} // namespace abseil
} // namespace tidy
} // namespace clang
