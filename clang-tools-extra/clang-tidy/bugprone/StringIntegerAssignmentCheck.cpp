//===--- StringIntegerAssignmentCheck.cpp - clang-tidy---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
              hasTemplateArgument(0, refersToType(qualType().bind("type"))))))),
          hasArgument(1,
                      ignoringImpCasts(expr(hasType(isInteger()),
                                            unless(hasType(isAnyCharacter())))
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
