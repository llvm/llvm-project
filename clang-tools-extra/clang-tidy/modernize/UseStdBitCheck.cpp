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

UseStdBitCheck::UseStdBitCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void UseStdBitCheck::registerMatchers(MatchFinder *Finder) {
  const auto MakeBinaryOperatorMatcher = [](auto Op) {
    return [=](const auto &LHS, const auto &RHS) {
      return binaryOperator(hasOperatorName(Op),
                            hasLHS(ignoringParenImpCasts(LHS)),
                            hasRHS(ignoringParenImpCasts(RHS)));
    };
  };
  const auto MakeCommutativeBinaryOperatorMatcher = [](auto Op) {
    return [=](const auto &LHS, const auto &RHS) {
      return binaryOperator(
          hasOperatorName(Op),
          hasOperands(ignoringParenImpCasts(LHS), ignoringParenImpCasts(RHS)));
    };
  };

  const auto LogicalAnd = MakeCommutativeBinaryOperatorMatcher("&&");
  const auto Sub = MakeBinaryOperatorMatcher("-");
  const auto BitwiseAnd = MakeCommutativeBinaryOperatorMatcher("&");
  const auto CmpNot = MakeCommutativeBinaryOperatorMatcher("!=");
  const auto CmpGt = MakeBinaryOperatorMatcher(">");
  const auto CmpGte = MakeBinaryOperatorMatcher(">=");
  const auto CmpLt = MakeBinaryOperatorMatcher("<");
  const auto CmpLte = MakeBinaryOperatorMatcher("<=");

  const auto Literal0 = integerLiteral(equals(0));
  const auto Literal1 = integerLiteral(equals(1));

  const auto LogicalNot = [](const auto &Expr) {
    return unaryOperator(hasOperatorName("!"),
                         hasUnaryOperand(ignoringParenImpCasts(Expr)));
  };

  const auto IsNonNull = [=](const auto &Expr) {
    return anyOf(Expr, CmpNot(Expr, Literal0), CmpGt(Expr, Literal0),
                 CmpGte(Expr, Literal1), CmpLt(Literal0, Expr),
                 CmpLte(Literal1, Expr));
  };
  const auto BindDeclRef = [](StringRef Name) {
    return declRefExpr(
        to(varDecl(hasType(isUnsignedInteger())).bind(Name.str())));
  };
  const auto BoundDeclRef = [](StringRef Name) {
    return declRefExpr(to(varDecl(equalsBoundNode(Name.str()))));
  };

  // Determining if an integer is a power of 2 with following pattern:
  // has_one_bit(v) = v && !(v & (v - 1));
  Finder->addMatcher(
      LogicalAnd(IsNonNull(BindDeclRef("v")),
                 LogicalNot(BitwiseAnd(
                     BoundDeclRef("v"),
                     Sub(BoundDeclRef("v"), integerLiteral(equals(1))))))
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

  auto Diag =
      diag(MatchedExpr->getBeginLoc(), "use 'std::has_one_bit' instead");
  if (auto R = MatchedExpr->getSourceRange();
      !R.getBegin().isMacroID() && !R.getEnd().isMacroID()) {
    Diag << FixItHint::CreateReplacement(
                MatchedExpr->getSourceRange(),
                ("std::has_one_bit(" + MatchedVarDecl->getName() + ")").str())
         << IncludeInserter.createIncludeInsertion(
                Source.getFileID(MatchedExpr->getBeginLoc()), "<bit>");
  }
}

} // namespace clang::tidy::modernize
