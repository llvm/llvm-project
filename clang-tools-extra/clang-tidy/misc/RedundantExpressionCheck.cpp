//===--- RedundantExpressionCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantExpressionCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

static const char KnownBannedMacroNames[] =
    "EAGAIN;EWOULDBLOCK;SIGCLD;SIGCHLD;";

static bool areEquivalentNameSpecifier(const NestedNameSpecifier *Left,
                                       const NestedNameSpecifier *Right) {
  llvm::FoldingSetNodeID LeftID, RightID;
  Left->Profile(LeftID);
  Right->Profile(RightID);
  return LeftID == RightID;
}

static bool areEquivalentExpr(const Expr *Left, const Expr *Right) {
  if (!Left || !Right)
    return !Left && !Right;

  Left = Left->IgnoreParens();
  Right = Right->IgnoreParens();

  // Compare classes.
  if (Left->getStmtClass() != Right->getStmtClass())
    return false;

  // Compare children.
  Expr::const_child_iterator LeftIter = Left->child_begin();
  Expr::const_child_iterator RightIter = Right->child_begin();
  while (LeftIter != Left->child_end() && RightIter != Right->child_end()) {
    if (!areEquivalentExpr(dyn_cast<Expr>(*LeftIter),
                           dyn_cast<Expr>(*RightIter)))
      return false;
    ++LeftIter;
    ++RightIter;
  }
  if (LeftIter != Left->child_end() || RightIter != Right->child_end())
    return false;

  // Perform extra checks.
  switch (Left->getStmtClass()) {
  default:
    return false;

  case Stmt::CharacterLiteralClass:
    return cast<CharacterLiteral>(Left)->getValue() ==
           cast<CharacterLiteral>(Right)->getValue();
  case Stmt::IntegerLiteralClass: {
    llvm::APInt LeftLit = cast<IntegerLiteral>(Left)->getValue();
    llvm::APInt RightLit = cast<IntegerLiteral>(Right)->getValue();
    return LeftLit.getBitWidth() == RightLit.getBitWidth() &&
           LeftLit == RightLit;
  }
  case Stmt::FloatingLiteralClass:
    return cast<FloatingLiteral>(Left)->getValue().bitwiseIsEqual(
        cast<FloatingLiteral>(Right)->getValue());
  case Stmt::StringLiteralClass:
    return cast<StringLiteral>(Left)->getBytes() ==
           cast<StringLiteral>(Right)->getBytes();

  case Stmt::DependentScopeDeclRefExprClass:
    if (cast<DependentScopeDeclRefExpr>(Left)->getDeclName() !=
        cast<DependentScopeDeclRefExpr>(Right)->getDeclName())
      return false;
    return areEquivalentNameSpecifier(
        cast<DependentScopeDeclRefExpr>(Left)->getQualifier(),
        cast<DependentScopeDeclRefExpr>(Right)->getQualifier());
  case Stmt::DeclRefExprClass:
    return cast<DeclRefExpr>(Left)->getDecl() ==
           cast<DeclRefExpr>(Right)->getDecl();
  case Stmt::MemberExprClass:
    return cast<MemberExpr>(Left)->getMemberDecl() ==
           cast<MemberExpr>(Right)->getMemberDecl();

  case Stmt::CStyleCastExprClass:
    return cast<CStyleCastExpr>(Left)->getTypeAsWritten() ==
           cast<CStyleCastExpr>(Right)->getTypeAsWritten();

  case Stmt::CallExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ArraySubscriptExprClass:
    return true;

  case Stmt::UnaryOperatorClass:
    if (cast<UnaryOperator>(Left)->isIncrementDecrementOp())
      return false;
    return cast<UnaryOperator>(Left)->getOpcode() ==
           cast<UnaryOperator>(Right)->getOpcode();
  case Stmt::BinaryOperatorClass:
    return cast<BinaryOperator>(Left)->getOpcode() ==
           cast<BinaryOperator>(Right)->getOpcode();
  }
}

AST_MATCHER(BinaryOperator, operandsAreEquivalent) {
  return areEquivalentExpr(Node.getLHS(), Node.getRHS());
}

AST_MATCHER(ConditionalOperator, expressionsAreEquivalent) {
  return areEquivalentExpr(Node.getTrueExpr(), Node.getFalseExpr());
}

AST_MATCHER(CallExpr, parametersAreEquivalent) {
  return Node.getNumArgs() == 2 &&
         areEquivalentExpr(Node.getArg(0), Node.getArg(1));
}

AST_MATCHER(BinaryOperator, binaryOperatorIsInMacro) {
  return Node.getOperatorLoc().isMacroID();
}

AST_MATCHER(ConditionalOperator, conditionalOperatorIsInMacro) {
  return Node.getQuestionLoc().isMacroID() || Node.getColonLoc().isMacroID();
}

AST_MATCHER(Expr, isMacro) { return Node.getExprLoc().isMacroID(); }

AST_MATCHER_P(Expr, expandedByMacro, std::set<std::string>, Names) {
  const SourceManager &SM = Finder->getASTContext().getSourceManager();
  const LangOptions &LO = Finder->getASTContext().getLangOpts();
  SourceLocation Loc = Node.getExprLoc();
  while (Loc.isMacroID()) {
    std::string MacroName = Lexer::getImmediateMacroName(Loc, SM, LO);
    if (Names.find(MacroName) != Names.end())
      return true;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }
  return false;
}

void RedundantExpressionCheck::registerMatchers(MatchFinder *Finder) {
  const auto AnyLiteralExpr = ignoringParenImpCasts(
      anyOf(cxxBoolLiteral(), characterLiteral(), integerLiteral()));

  std::vector<std::string> MacroNames =
      utils::options::parseStringList(KnownBannedMacroNames);
  std::set<std::string> Names(MacroNames.begin(), MacroNames.end());

  const auto BannedIntegerLiteral = integerLiteral(expandedByMacro(Names));

  Finder->addMatcher(
      binaryOperator(anyOf(hasOperatorName("-"), hasOperatorName("/"),
                           hasOperatorName("%"), hasOperatorName("|"),
                           hasOperatorName("&"), hasOperatorName("^"),
                           matchers::isComparisonOperator(),
                           hasOperatorName("&&"), hasOperatorName("||"),
                           hasOperatorName("=")),
                     operandsAreEquivalent(),
                     // Filter noisy false positives.
                     unless(isInTemplateInstantiation()),
                     unless(binaryOperatorIsInMacro()),
                     unless(hasType(realFloatingPointType())),
                     unless(hasEitherOperand(hasType(realFloatingPointType()))),
                     unless(hasLHS(AnyLiteralExpr)),
                     unless(hasDescendant(BannedIntegerLiteral)))
          .bind("binary"),
      this);

  Finder->addMatcher(
      conditionalOperator(expressionsAreEquivalent(),
                          // Filter noisy false positives.
                          unless(conditionalOperatorIsInMacro()),
                          unless(hasTrueExpression(AnyLiteralExpr)),
                          unless(isInTemplateInstantiation()))
          .bind("cond"),
      this);

  Finder->addMatcher(
      cxxOperatorCallExpr(
          anyOf(
              hasOverloadedOperatorName("-"), hasOverloadedOperatorName("/"),
              hasOverloadedOperatorName("%"), hasOverloadedOperatorName("|"),
              hasOverloadedOperatorName("&"), hasOverloadedOperatorName("^"),
              hasOverloadedOperatorName("=="), hasOverloadedOperatorName("!="),
              hasOverloadedOperatorName("<"), hasOverloadedOperatorName("<="),
              hasOverloadedOperatorName(">"), hasOverloadedOperatorName(">="),
              hasOverloadedOperatorName("&&"), hasOverloadedOperatorName("||"),
              hasOverloadedOperatorName("=")),
          parametersAreEquivalent(),
          // Filter noisy false positives.
          unless(isMacro()), unless(isInTemplateInstantiation()))
          .bind("call"),
      this);
}

void RedundantExpressionCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binary"))
    diag(BinOp->getOperatorLoc(), "both side of operator are equivalent");
  if (const auto *CondOp = Result.Nodes.getNodeAs<ConditionalOperator>("cond"))
    diag(CondOp->getColonLoc(), "'true' and 'false' expression are equivalent");
  if (const auto *Call = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("call"))
    diag(Call->getOperatorLoc(),
         "both side of overloaded operator are equivalent");
}

} // namespace misc
} // namespace tidy
} // namespace clang
