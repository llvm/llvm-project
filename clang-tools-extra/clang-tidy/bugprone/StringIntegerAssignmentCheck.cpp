//===----------------------------------------------------------------------===//
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

namespace clang::tidy::bugprone {

void StringIntegerAssignmentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxOperatorCallExpr(
          hasAnyOverloadedOperatorName("=", "+="),
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

class CharExpressionDetector {
public:
  CharExpressionDetector(QualType CharType, const ASTContext &Ctx)
      : CharType(CharType), Ctx(Ctx) {}

  bool isLikelyCharExpression(const Expr *E) const {
    if (isCharTyped(E))
      return true;

    if (const auto *BinOp = dyn_cast<BinaryOperator>(E)) {
      const auto *LHS = BinOp->getLHS()->IgnoreParenImpCasts();
      const auto *RHS = BinOp->getRHS()->IgnoreParenImpCasts();
      // Handle both directions, e.g. `'a' + (i % 26)` and `(i % 26) + 'a'`.
      if (BinOp->isAdditiveOp() || BinOp->isBitwiseOp())
        return handleBinaryOp(BinOp->getOpcode(), LHS, RHS) ||
               handleBinaryOp(BinOp->getOpcode(), RHS, LHS);
      // Except in the case of '%'.
      if (BinOp->getOpcode() == BO_Rem)
        return handleBinaryOp(BinOp->getOpcode(), LHS, RHS);
      return false;
    }

    // Ternary where at least one branch is a likely char expression, e.g.
    //    i < 265 ? i : ' '
    if (const auto *CondOp = dyn_cast<AbstractConditionalOperator>(E))
      return isLikelyCharExpression(
                 CondOp->getFalseExpr()->IgnoreParenImpCasts()) ||
             isLikelyCharExpression(
                 CondOp->getTrueExpr()->IgnoreParenImpCasts());
    return false;
  }

private:
  bool handleBinaryOp(clang::BinaryOperatorKind Opcode, const Expr *const LHS,
                      const Expr *const RHS) const {
    // <char_expr> <op> <char_expr> (c++ integer promotion rules make this an
    // int), e.g.
    //    'a' + c
    if (isCharTyped(LHS) && isCharTyped(RHS))
      return true;

    // <expr> & <char_valued_constant> or <expr> % <char_valued_constant>, e.g.
    //    i & 0xff
    if ((Opcode == BO_And || Opcode == BO_Rem) && isCharValuedConstant(RHS))
      return true;

    // <char_expr> | <char_valued_constant>, e.g.
    //    c | 0x80
    if (Opcode == BO_Or && isCharTyped(LHS) && isCharValuedConstant(RHS))
      return true;

    // <char_constant> + <likely_char_expr>, e.g.
    //    'a' + (i % 26)
    if (Opcode == BO_Add)
      return isCharConstant(LHS) && isLikelyCharExpression(RHS);

    return false;
  }

  // Returns true if `E` is an character constant.
  bool isCharConstant(const Expr *E) const {
    return isCharTyped(E) && isCharValuedConstant(E);
  };

  // Returns true if `E` is an integer constant which fits in `CharType`.
  bool isCharValuedConstant(const Expr *E) const {
    if (E->isInstantiationDependent())
      return false;
    Expr::EvalResult EvalResult;
    if (!E->EvaluateAsInt(EvalResult, Ctx, Expr::SE_AllowSideEffects))
      return false;
    return EvalResult.Val.getInt().getActiveBits() <= Ctx.getTypeSize(CharType);
  };

  // Returns true if `E` has the right character type.
  bool isCharTyped(const Expr *E) const {
    return E->getType().getCanonicalType().getTypePtr() ==
           CharType.getTypePtr();
  };

  const QualType CharType;
  const ASTContext &Ctx;
};

void StringIntegerAssignmentCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Argument = Result.Nodes.getNodeAs<Expr>("expr");
  const auto CharType =
      Result.Nodes.getNodeAs<QualType>("type")->getCanonicalType();
  SourceLocation Loc = Argument->getBeginLoc();

  // Try to detect a few common expressions to reduce false positives.
  if (CharExpressionDetector(CharType, *Result.Context)
          .isLikelyCharExpression(Argument))
    return;

  auto Diag =
      diag(Loc, "an integer is interpreted as a character code when assigning "
                "it to a string; if this is intended, cast the integer to the "
                "appropriate character type; if you want a string "
                "representation, use the appropriate conversion facility");

  if (Loc.isMacroID())
    return;

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

} // namespace clang::tidy::bugprone
