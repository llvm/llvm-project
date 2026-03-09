//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdBitCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void UseStdBitCheck::registerMatchers(MatchFinder *Finder) {
  const auto makeBinaryOperatorMatcher = [](auto Op) {
    return [=](auto LHS, auto RHS) {
      return binaryOperator(
          hasOperatorName(Op),
          hasOperands(ignoringParenImpCasts(LHS), ignoringParenImpCasts(RHS)));
    };
  };

  const auto logicalAnd = makeBinaryOperatorMatcher("&&");
  const auto sub = makeBinaryOperatorMatcher("-");
  const auto bitwiseAnd = makeBinaryOperatorMatcher("&");
  const auto cmpNot = makeBinaryOperatorMatcher("!=");
  const auto cmpGt = makeBinaryOperatorMatcher(">");

  const auto logicalNot = [](auto Expr) {
    return unaryOperator(hasOperatorName("!"),
                         hasUnaryOperand(ignoringParenImpCasts(Expr)));
  };

  const auto isNonNull = [=](auto Expr) {
    return anyOf(Expr, cmpNot(Expr, integerLiteral(equals(0))),
                 cmpGt(Expr, integerLiteral(equals(0))));
  };
  const auto bindDeclRef = [](auto Name) {
    return declRefExpr(to(varDecl(hasType(isUnsignedInteger())).bind(Name)));
  };
  const auto boundDeclRef = [](auto Name) {
    return declRefExpr(to(varDecl(equalsBoundNode(Name))));
  };

  // https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
  // has_one_bit(v) = v && !(v & (v - 1));
  Finder->addMatcher(
      logicalAnd(isNonNull(bindDeclRef("v")),
                 logicalNot(bitwiseAnd(
                     boundDeclRef("v"),
                     sub(boundDeclRef("v"), integerLiteral(equals(1))))))
          .bind("expr"),
      this);
}

void UseStdBitCheck::registerPPCallbacks(const SourceManager &SM,
                                         Preprocessor *PP,
                                         Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseStdBitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

void UseStdBitCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();

  const auto *MatchedVarDecl = Result.Nodes.getNodeAs<VarDecl>("v");
  const auto *MatchedExpr = Result.Nodes.getNodeAs<BinaryOperator>("expr");

  diag(MatchedExpr->getBeginLoc(), "use std::has_one_bit instead")
      << MatchedVarDecl->getName()
      << FixItHint::CreateReplacement(
             MatchedExpr->getSourceRange(),
             ("std::has_one_bit(" + MatchedVarDecl->getName() + ")").str())
      << IncludeInserter.createIncludeInsertion(
             Source.getFileID(MatchedExpr->getBeginLoc()), "<bit>");
}

} // namespace clang::tidy::modernize
