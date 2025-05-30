//===--- BoolBitwiseOperationCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoolBitwiseOperationCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <array>
#include <utility>
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {
namespace {

constexpr std::array<std::pair<llvm::StringRef, llvm::StringRef>, 8U>
    OperatorsTransformation{{{"|", "||"},
                             {"|=", "||"},
                             {"&", "&&"},
                             {"&=", "&&"},
                             {"bitand", "and"},
                             {"and_eq", "and"},
                             {"bitor", "or"},
                             {"or_eq", "or"}}};

llvm::StringRef translate(llvm::StringRef Value) {
  for (const auto &[Bitwise, Logical] : OperatorsTransformation) {
    if (Value == Bitwise)
      return Logical;
  }

  return {};
}
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(binaryOperator(
        unless(isExpansionInSystemHeader()),
        hasAnyOperatorName("|", "&", "|=", "&="),
        hasEitherOperand(expr(ignoringImpCasts(hasType(booleanType())))),
        optionally(hasEitherOperand(
            expr(ignoringImpCasts(hasType(isVolatileQualified())))
                .bind("vol"))),
        optionally(hasAncestor(
            binaryOperator().bind("p"))),
        optionally(hasRHS(ignoringParenCasts(
            binaryOperator().bind("r")))),
        optionally(hasLHS(ignoringParenCasts(
            declRefExpr().bind("l")
        )))
        )
    .bind("op"), this);
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<BinaryOperator>("op");

  auto Diag = diag(MatchedExpr->getOperatorLoc(), "use logical operator instead of bitwise one for bool");

  const auto *VolatileOperand = Result.Nodes.getNodeAs<Expr>("vol");
  if (VolatileOperand)
    return;

  SourceLocation Loc = MatchedExpr->getOperatorLoc();

  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  Loc = Result.SourceManager->getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  if (TokenRange.isInvalid())
    return;

  StringRef Spelling = Lexer::getSourceText(TokenRange, *Result.SourceManager,
                                            Result.Context->getLangOpts());
  StringRef TranslatedSpelling = translate(Spelling);

  if (TranslatedSpelling.empty())
    return;

  std::string FixSpelling = TranslatedSpelling.str();

  FixItHint InsertEqual, ReplaceOperator, InsertBrace1, InsertBrace2;
  if (MatchedExpr->isCompoundAssignmentOp()) {
    const auto *DelcRefLHS = Result.Nodes.getNodeAs<DeclRefExpr>("l");
    if (!DelcRefLHS)
      return;
    const SourceLocation LocLHS = DelcRefLHS->getEndLoc();
    if (LocLHS.isInvalid() || LocLHS.isMacroID())
      return;
    const SourceLocation InsertLoc = clang::Lexer::getLocForEndOfToken(
        LocLHS, 0, *Result.SourceManager,
        Result.Context->getLangOpts());
    if (InsertLoc.isInvalid() || InsertLoc.isMacroID()) {
      return;
    }
    InsertEqual = FixItHint::CreateInsertion(InsertLoc,
                                        " = " + DelcRefLHS->getDecl()->getNameAsString());
  }
  ReplaceOperator = FixItHint::CreateReplacement(TokenRange, FixSpelling);

  std::optional<BinaryOperatorKind> ParentOpcode;
  if (const auto *Parent = Result.Nodes.getNodeAs<BinaryOperator>("p"); Parent)
    ParentOpcode = Parent->getOpcode();

  const auto *RHS = Result.Nodes.getNodeAs<BinaryOperator>("r");
  std::optional<BinaryOperatorKind> RHSOpcode;
  if (RHS)
    RHSOpcode = RHS->getOpcode();

  const BinaryOperator* SurroundedExpr = nullptr;
  if ((MatchedExpr->getOpcode() == BO_Or && ParentOpcode.has_value() && *ParentOpcode == BO_LAnd) ||
      (MatchedExpr->getOpcode() == BO_And && ParentOpcode.has_value() && *ParentOpcode == BO_Xor)) {
    SurroundedExpr = MatchedExpr;
  } else if (MatchedExpr->getOpcode() == BO_AndAssign && RHSOpcode.has_value() && *RHSOpcode == BO_LOr) {
    SurroundedExpr = RHS;
  }

  if (SurroundedExpr) {
    const SourceLocation InsertFirstLoc = SurroundedExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc =
        clang::Lexer::getLocForEndOfToken(SurroundedExpr->getEndLoc(), 0, *Result.SourceManager,
        Result.Context->getLangOpts());
    if (InsertFirstLoc.isInvalid() || InsertFirstLoc.isMacroID() || InsertSecondLoc.isInvalid() || InsertSecondLoc.isMacroID())
      return;
    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  Diag << InsertEqual << ReplaceOperator << InsertBrace1 << InsertBrace2;
}

} // namespace clang::tidy::performance
