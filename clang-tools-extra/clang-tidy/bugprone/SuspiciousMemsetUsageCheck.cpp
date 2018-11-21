//===--- SuspiciousMemsetUsageCheck.cpp - clang-tidy-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SuspiciousMemsetUsageCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void SuspiciousMemsetUsageCheck::registerMatchers(MatchFinder *Finder) {
  // Note: void *memset(void *buffer, int fill_char, size_t byte_count);
  // Look for memset(x, '0', z). Probably memset(x, 0, z) was intended.
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::memset"))),
          hasArgument(1, characterLiteral(equals(static_cast<unsigned>('0')))
                             .bind("char-zero-fill")),
          unless(
              eachOf(hasArgument(0, anyOf(hasType(pointsTo(isAnyCharacter())),
                                          hasType(arrayType(hasElementType(
                                              isAnyCharacter()))))),
                     isInTemplateInstantiation()))),
      this);

  // Look for memset with an integer literal in its fill_char argument.
  // Will check if it gets truncated.
  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::memset"))),
                              hasArgument(1, integerLiteral().bind("num-fill")),
                              unless(isInTemplateInstantiation())),
                     this);

  // Look for memset(x, y, 0) as that is most likely an argument swap.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::memset"))),
               unless(hasArgument(1, anyOf(characterLiteral(equals(
                                               static_cast<unsigned>('0'))),
                                           integerLiteral()))),
               unless(isInTemplateInstantiation()))
          .bind("call"),
      this);
}

void SuspiciousMemsetUsageCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *CharZeroFill =
          Result.Nodes.getNodeAs<CharacterLiteral>("char-zero-fill")) {
    // Case 1: fill_char of memset() is a character '0'. Probably an
    // integer zero was intended.

    SourceRange CharRange = CharZeroFill->getSourceRange();
    auto Diag =
        diag(CharZeroFill->getBeginLoc(), "memset fill value is char '0', "
                                          "potentially mistaken for int 0");

    // Only suggest a fix if no macros are involved.
    if (CharRange.getBegin().isMacroID())
      return;
    Diag << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(CharRange), "0");
  }

  else if (const auto *NumFill =
               Result.Nodes.getNodeAs<IntegerLiteral>("num-fill")) {
    // Case 2: fill_char of memset() is larger in size than an unsigned char
    // so it gets truncated during conversion.

    const auto UCharMax = (1 << Result.Context->getCharWidth()) - 1;
    Expr::EvalResult EVResult;
    if (!NumFill->EvaluateAsInt(EVResult, *Result.Context))
      return;

    llvm::APSInt NumValue = EVResult.Val.getInt();
    if (NumValue >= 0 && NumValue <= UCharMax)
      return;

    diag(NumFill->getBeginLoc(), "memset fill value is out of unsigned "
                                 "character range, gets truncated");
  }

  else if (const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call")) {
    // Case 3: byte_count of memset() is zero. This is most likely an
    // argument swap.

    const Expr *FillChar = Call->getArg(1);
    const Expr *ByteCount = Call->getArg(2);

    // Return if `byte_count` is not zero at compile time.
    Expr::EvalResult Value2;
    if (ByteCount->isValueDependent() ||
        !ByteCount->EvaluateAsInt(Value2, *Result.Context) ||
        Value2.Val.getInt() != 0)
      return;

    // Return if `fill_char` is known to be zero or negative at compile
    // time. In these cases, swapping the args would be a nop, or
    // introduce a definite bug. The code is likely correct.
    Expr::EvalResult EVResult;
    if (!FillChar->isValueDependent() &&
        FillChar->EvaluateAsInt(EVResult, *Result.Context)) {
      llvm::APSInt Value1 = EVResult.Val.getInt();
      if (Value1 == 0 || Value1.isNegative())
        return;
    }

    // `byte_count` is known to be zero at compile time, and `fill_char` is
    // either not known or known to be a positive integer. Emit a warning
    // and fix-its to swap the arguments.
    auto D = diag(Call->getBeginLoc(),
                  "memset of size zero, potentially swapped arguments");
    StringRef RHSString = tooling::fixit::getText(*ByteCount, *Result.Context);
    StringRef LHSString = tooling::fixit::getText(*FillChar, *Result.Context);
    if (LHSString.empty() || RHSString.empty())
      return;

    D << tooling::fixit::createReplacement(*FillChar, RHSString)
      << tooling::fixit::createReplacement(*ByteCount, LHSString);
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
