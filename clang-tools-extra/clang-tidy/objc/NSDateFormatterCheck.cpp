//===--- NSDateFormatterCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NSDateFormatterCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::objc {

void NSDateFormatterCheck::registerMatchers(MatchFinder *Finder) {
  // Adding matchers.

  Finder->addMatcher(
      objcMessageExpr(hasSelector("setDateFormat:"),
                      hasReceiverType(asString("NSDateFormatter *")),
                      hasArgument(0, ignoringImpCasts(
                                         objcStringLiteral().bind("str_lit")))),
      this);
}

static char ValidDatePatternChars[] = {
    'G', 'y', 'Y', 'u', 'U', 'r', 'Q', 'q', 'M', 'L', 'I', 'w', 'W', 'd',
    'D', 'F', 'g', 'E', 'e', 'c', 'a', 'b', 'B', 'h', 'H', 'K', 'k', 'j',
    'J', 'C', 'm', 's', 'S', 'A', 'z', 'Z', 'O', 'v', 'V', 'X', 'x'};

// Checks if the string pattern used as a date format specifier is valid.
// A string pattern is valid if all the letters(a-z, A-Z) in it belong to the
// set of reserved characters. See:
// https://www.unicode.org/reports/tr35/tr35.html#Invalid_Patterns
bool isValidDatePattern(StringRef Pattern) {
  return llvm::all_of(Pattern, [](const auto &PatternChar) {
    return !isalpha(PatternChar) ||
           llvm::is_contained(ValidDatePatternChars, PatternChar);
  });
}

// Checks if the string pattern used as a date format specifier contains
// any incorrect pattern and reports it as a warning.
// See: http://www.unicode.org/reports/tr35/tr35-dates.html#Date_Format_Patterns
void NSDateFormatterCheck::check(const MatchFinder::MatchResult &Result) {
  // Callback implementation.
  const auto *StrExpr = Result.Nodes.getNodeAs<ObjCStringLiteral>("str_lit");
  const StringLiteral *SL = cast<ObjCStringLiteral>(StrExpr)->getString();
  StringRef SR = SL->getString();

  if (!isValidDatePattern(SR)) {
    diag(StrExpr->getExprLoc(), "invalid date format specifier");
  }

  if (SR.contains('y') && SR.contains('w') && !SR.contains('Y')) {
    diag(StrExpr->getExprLoc(),
         "use of calendar year (y) with week of the year (w); "
         "did you mean to use week-year (Y) instead?");
  }
  if (SR.contains('F')) {
    if (!(SR.contains('e') || SR.contains('E'))) {
      diag(StrExpr->getExprLoc(),
           "day of week in month (F) used without day of the week (e or E); "
           "did you forget e (or E) in the format string?");
    }
    if (!SR.contains('M')) {
      diag(StrExpr->getExprLoc(),
           "day of week in month (F) used without the month (M); "
           "did you forget M in the format string?");
    }
  }
  if (SR.contains('W') && !SR.contains('M')) {
    diag(StrExpr->getExprLoc(), "Week of Month (W) used without the month (M); "
                                "did you forget M in the format string?");
  }
  if (SR.contains('Y') && SR.contains('Q') && !SR.contains('y')) {
    diag(StrExpr->getExprLoc(),
         "use of week year (Y) with quarter number (Q); "
         "did you mean to use calendar year (y) instead?");
  }
  if (SR.contains('Y') && SR.contains('M') && !SR.contains('y')) {
    diag(StrExpr->getExprLoc(),
         "use of week year (Y) with month (M); "
         "did you mean to use calendar year (y) instead?");
  }
  if (SR.contains('Y') && SR.contains('D') && !SR.contains('y')) {
    diag(StrExpr->getExprLoc(),
         "use of week year (Y) with day of the year (D); "
         "did you mean to use calendar year (y) instead?");
  }
  if (SR.contains('Y') && SR.contains('W') && !SR.contains('y')) {
    diag(StrExpr->getExprLoc(),
         "use of week year (Y) with week of the month (W); "
         "did you mean to use calendar year (y) instead?");
  }
  if (SR.contains('Y') && SR.contains('F') && !SR.contains('y')) {
    diag(StrExpr->getExprLoc(),
         "use of week year (Y) with day of the week in month (F); "
         "did you mean to use calendar year (y) instead?");
  }
}

} // namespace clang::tidy::objc
