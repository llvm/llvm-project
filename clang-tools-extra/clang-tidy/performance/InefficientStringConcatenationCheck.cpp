//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientStringConcatenationCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void InefficientStringConcatenationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

InefficientStringConcatenationCheck::InefficientStringConcatenationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", false)) {}

void InefficientStringConcatenationCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto BasicStringType =
      hasType(qualType(hasUnqualifiedDesugaredType(recordType(
          hasDeclaration(cxxRecordDecl(hasName("::std::basic_string")))))));

  const auto BasicStringPlusOperator = cxxOperatorCallExpr(
      hasOverloadedOperatorName("+"),
      hasAnyArgument(ignoringImpCasts(declRefExpr(BasicStringType))));

  const auto PlusOperator =
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("+"),
          hasAnyArgument(ignoringImpCasts(declRefExpr(BasicStringType))),
          hasDescendant(BasicStringPlusOperator))
          .bind("plusOperator");

  const auto AssignOperator =
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("="),
          hasArgument(0, ignoringParenImpCasts(
                             declRefExpr(BasicStringType,
                                         hasDeclaration(decl().bind("lhsStrT")))
                                 .bind("lhsStr"))),
          hasArgument(
              1, expr(hasDescendant(BasicStringPlusOperator)).bind("rhsExpr")))
          .bind("assign");

  if (StrictMode) {
    Finder->addMatcher(cxxOperatorCallExpr(anyOf(AssignOperator, PlusOperator)),
                       this);
  } else {
    Finder->addMatcher(cxxOperatorCallExpr(anyOf(AssignOperator, PlusOperator),
                                           hasAncestor(stmt(anyOf(
                                               cxxForRangeStmt(), whileStmt(),
                                               forStmt(), doStmt())))),
                       this);
  }
}

static const Expr *strip(const Expr *E) {
  while (true) {
    E = E->IgnoreParenImpCasts();
    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E)) {
      E = ICE->getSubExpr();
    } else if (const auto *M = dyn_cast<MaterializeTemporaryExpr>(E)) {
      E = M->getSubExpr();
    }

    else if (const auto *B = dyn_cast<CXXBindTemporaryExpr>(E)) {
      E = B->getSubExpr();
    } else {
      break;
    }
  }
  return E;
}

static void collectOperands(const Expr *E, SmallVector<const Expr *> &Ops) {
  E = strip(E);
  E = E->IgnoreParenImpCasts();

  if (const auto *BinOp = dyn_cast<BinaryOperator>(E)) {
    if (BinOp->getOpcode() == BO_Add) {
      collectOperands(BinOp->getLHS(), Ops);
      collectOperands(BinOp->getRHS(), Ops);
      return;
    }
  }

  if (const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (OpCall->getOperator() == OO_Plus) {
      collectOperands(OpCall->getArg(0), Ops);
      collectOperands(OpCall->getArg(1), Ops);
      return;
    }
  }

  Ops.push_back(E);
}

static bool isSameLhs(const Expr *E, const DeclRefExpr *Lhs) {
  E = strip(E)->IgnoreParenImpCasts();
  auto *DR = dyn_cast<DeclRefExpr>(E);
  return DR && DR->getDecl() == Lhs->getDecl();
}

void InefficientStringConcatenationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *LhsStr = Result.Nodes.getNodeAs<DeclRefExpr>("lhsStr");
  const auto *PlusOperator =
      Result.Nodes.getNodeAs<CXXOperatorCallExpr>("plusOperator");
  const auto *Assign = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("assign");
  const auto *RhsExpr = Result.Nodes.getNodeAs<Expr>("rhsExpr");

  const char *DiagMsg =
      "string concatenation results in allocation of unnecessary temporary "
      "strings; consider using 'operator+=' or 'string::append()' instead";

  if (Assign && LhsStr && RhsExpr) {
    const auto &SM = *Result.SourceManager;
    const auto &LO = Result.Context->getLangOpts();

    SmallVector<const Expr *> Operands;
    collectOperands(RhsExpr, Operands);

    size_t LhsPosition = -1;
    for (size_t i = 0; i < Operands.size(); ++i)
      if (isSameLhs(Operands[i], LhsStr)) {
        LhsPosition = i;
        break;
      }
    // skip if the LHS string is not the leftmost operand.
    if (LhsPosition != 0)
      return;

    auto ReplacementText =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(LhsStr->getSourceRange()), SM, LO)
            .str();

    if (Operands.size() > 2) {
      for (size_t i = 0; i < Operands.size(); ++i) {
        if (i == LhsPosition)
          continue;
        auto OpText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(Operands[i]->getSourceRange()), SM,
            LO);
        ReplacementText += ".append(" + OpText.str() + ")";
      }
    } else {
      auto RhsText = Lexer::getSourceText(
          CharSourceRange::getTokenRange(
              Operands[LhsPosition == 0 ? 1 : 0]->getSourceRange()),
          SM, LO);
      ReplacementText += " += " + RhsText.str();
    }

    diag(Assign->getExprLoc(), DiagMsg) << FixItHint::CreateReplacement(
        Assign->getSourceRange(), ReplacementText);

  } else {
    if (LhsStr)
      diag(LhsStr->getExprLoc(), DiagMsg);
    else if (PlusOperator)
      diag(PlusOperator->getExprLoc(), DiagMsg);
  }
}

} // namespace clang::tidy::performance
