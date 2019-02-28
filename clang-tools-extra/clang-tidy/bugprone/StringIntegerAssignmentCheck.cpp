//===--- StringIntegerAssignmentCheck.cpp - clang-tidy---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringIntegerAssignmentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void StringIntegerAssignmentCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;
  Finder->addMatcher(
      cxxOperatorCallExpr(
          anyOf(hasOverloadedOperatorName("="),
                hasOverloadedOperatorName("+=")),
          callee(cxxMethodDecl(ofClass(classTemplateSpecializationDecl(
              hasName("::std::basic_string"),
              hasTemplateArgument(0, refersToType(hasCanonicalType(
                                         qualType().bind("type")))))))),
          hasArgument(
              1,
              ignoringImpCasts(
                  expr(hasType(isInteger()), unless(hasType(isAnyCharacter())),
                       // Ignore calls to tolower/toupper (see PR27723).
                       unless(callExpr(callee(functionDecl(
                           hasAnyName("tolower", "std::tolower", "toupper",
                                      "std::toupper"))))),
                       // Do not warn if assigning e.g. `CodePoint` to
                       // `basic_string<CodePoint>`
                       unless(hasType(qualType(
                           hasCanonicalType(equalsBoundNode("type"))))))
                      .bind("expr"))),
          unless(isInTemplateInstantiation())),
      this);
}

void StringIntegerAssignmentCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Argument = Result.Nodes.getNodeAs<Expr>("expr");
  SourceLocation Loc = Argument->getBeginLoc();

  auto Diag =
      diag(Loc, "an integer is interpreted as a character code when assigning "
                "it to a string; if this is intended, cast the integer to the "
                "appropriate character type; if you want a string "
                "representation, use the appropriate conversion facility");

  if (Loc.isMacroID())
    return;

  auto CharType = *Result.Nodes.getNodeAs<QualType>("type");
  bool IsWideCharType = CharType->isWideCharType();
  if (!CharType->isCharType() && !IsWideCharType)
    return;
  bool IsOneDigit = false;
  bool IsLiteral = false;
  if (const auto *Literal = dyn_cast<IntegerLiteral>(Argument)) {
    IsOneDigit = Literal->getValue().getLimitedValue() < 10;
    IsLiteral = true;
  }

  SourceLocation EndLoc = Lexer::getLocForEndOfToken(
      Argument->getEndLoc(), 0, *Result.SourceManager, getLangOpts());
  if (IsOneDigit) {
    Diag << FixItHint::CreateInsertion(Loc, IsWideCharType ? "L'" : "'")
         << FixItHint::CreateInsertion(EndLoc, "'");
    return;
  }
  if (IsLiteral) {
    Diag << FixItHint::CreateInsertion(Loc, IsWideCharType ? "L\"" : "\"")
         << FixItHint::CreateInsertion(EndLoc, "\"");
    return;
  }

  if (getLangOpts().CPlusPlus11) {
    Diag << FixItHint::CreateInsertion(Loc, IsWideCharType ? "std::to_wstring("
                                                           : "std::to_string(")
         << FixItHint::CreateInsertion(EndLoc, ")");
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
