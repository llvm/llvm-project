//===--- UseIntegerSignComparisonCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseIntegerSignComparisonCheck.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang::tidy::modernize {

UseIntegerSignComparisonCheck::UseIntegerSignComparisonCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()),
      IsQtApplication(Options.get("IsQtApplication", false)) {}

void UseIntegerSignComparisonCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IsQtApplication", IsQtApplication);
}

void UseIntegerSignComparisonCheck::registerMatchers(MatchFinder *Finder) {
  const auto SignedIntCastExpr = intCastExpression(true, "sIntCastExpression");
  const auto UnSignedIntCastExpr =
      intCastExpression(false, "uIntCastExpression");

  // Flag all operators "==", "<=", ">=", "<", ">", "!="
  // that are used between signed/unsigned
  const auto CompareOperator =
      expr(binaryOperator(hasAnyOperatorName("==", "<=", ">=", "<", ">", "!="),
                          anyOf(allOf(hasLHS(SignedIntCastExpr),
                                      hasRHS(UnSignedIntCastExpr)),
                                allOf(hasLHS(UnSignedIntCastExpr),
                                      hasRHS(SignedIntCastExpr)))))
          .bind("intComparison");

  Finder->addMatcher(CompareOperator, this);
}

BindableMatcher<clang::Stmt> UseIntegerSignComparisonCheck::intCastExpression(
    bool IsSigned, const std::string &CastBindName) const {
  auto IntTypeExpr = expr();
  if (IsSigned) {
    IntTypeExpr = expr(hasType(qualType(isInteger(), isSignedInteger())));
  } else {
    IntTypeExpr =
        expr(hasType(qualType(isInteger(), unless(isSignedInteger()))));
  }

  const auto ImplicitCastExpr =
      implicitCastExpr(hasSourceExpression(IntTypeExpr)).bind(CastBindName);

  const auto CStyleCastExpr = cStyleCastExpr(has(ImplicitCastExpr));
  const auto StaticCastExpr = cxxStaticCastExpr(has(ImplicitCastExpr));
  const auto FunctionalCastExpr = cxxFunctionalCastExpr(has(ImplicitCastExpr));

  return traverse(TK_AsIs, expr(anyOf(ImplicitCastExpr, CStyleCastExpr,
                                      StaticCastExpr, FunctionalCastExpr)));
}

std::string
UseIntegerSignComparisonCheck::parseOpCode(BinaryOperator::Opcode code) const {
  switch (code) {
  case BO_LT:
    return std::string("cmp_less");
  case BO_GT:
    return std::string("cmp_greater");
  case BO_LE:
    return std::string("cmp_less_equal");
  case BO_GE:
    return std::string("cmp_greater_equal");
  case BO_EQ:
    return std::string("cmp_equal");
  case BO_NE:
    return std::string("cmp_not_equal");
  default:
    return std::string();
  }
}

void UseIntegerSignComparisonCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseIntegerSignComparisonCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *SignedCastExpression =
      Result.Nodes.getNodeAs<ImplicitCastExpr>("sIntCastExpression");
  const auto *UnSignedCastExpression =
      Result.Nodes.getNodeAs<ImplicitCastExpr>("uIntCastExpression");
  assert(SignedCastExpression);
  assert(UnSignedCastExpression);

  // Ignore the match if we know that the signed int value is not negative.
  Expr::EvalResult EVResult;
  if (!SignedCastExpression->isValueDependent() &&
      SignedCastExpression->getSubExpr()->EvaluateAsInt(EVResult,
                                                        *Result.Context)) {
    llvm::APSInt SValue = EVResult.Val.getInt();
    if (SValue.isNonNegative())
      return;
  }

  const auto *BinaryOp =
      Result.Nodes.getNodeAs<BinaryOperator>("intComparison");
  if (BinaryOp == nullptr)
    return;

  auto OpCode = BinaryOp->getOpcode();
  const auto *LHS = BinaryOp->getLHS()->IgnoreParenImpCasts();
  const auto *RHS = BinaryOp->getRHS()->IgnoreParenImpCasts();
  if (LHS == nullptr || RHS == nullptr)
    return;

  StringRef LHSString(Lexer::getSourceText(
      CharSourceRange::getTokenRange(LHS->getSourceRange()),
      *Result.SourceManager, getLangOpts()));

  StringRef RHSString(Lexer::getSourceText(
      CharSourceRange::getTokenRange(RHS->getSourceRange()),
      *Result.SourceManager, getLangOpts()));

  DiagnosticBuilder Diag =
      diag(BinaryOp->getBeginLoc(),
           "comparison between 'signed' and 'unsigned' integers");

  if (!(getLangOpts().CPlusPlus17 && IsQtApplication) &&
      !getLangOpts().CPlusPlus20)
    return;

  std::string CmpNamespace;
  std::string CmpInclude;
  if (getLangOpts().CPlusPlus17 && IsQtApplication) {
    CmpInclude = "<QtCore/q20utility.h>";
    CmpNamespace = std::string("q20::") + parseOpCode(OpCode);
  }

  if (getLangOpts().CPlusPlus20) {
    CmpInclude = "<utility>";
    CmpNamespace = std::string("std::") + parseOpCode(OpCode);
  }

  // Use qt-use-integer-sign-comparison when C++17 is available and only for Qt
  // apps. Prefer modernize-use-integer-sign-comparison when C++20 is available!
  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(BinaryOp->getBeginLoc(),
                                     BinaryOp->getEndLoc()),
      StringRef(std::string(CmpNamespace) + std::string("(") +
                std::string(LHSString) + std::string(", ") +
                std::string(RHSString) + std::string(")")));

  // If there is no include for cmp_{*} functions, we'll add it.
  Diag << IncludeInserter.createIncludeInsertion(
      Result.SourceManager->getFileID(BinaryOp->getBeginLoc()),
      StringRef(CmpInclude));
}

} // namespace clang::tidy::modernize
