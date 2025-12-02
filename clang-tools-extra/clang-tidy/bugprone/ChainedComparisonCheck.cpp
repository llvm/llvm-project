//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ChainedComparisonCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {
static bool isExprAComparisonOperator(const Expr *E) {
  if (const auto *Op = dyn_cast_or_null<BinaryOperator>(E->IgnoreImplicit()))
    return Op->isComparisonOp();
  if (const auto *Op =
          dyn_cast_or_null<CXXOperatorCallExpr>(E->IgnoreImplicit()))
    return Op->isComparisonOp();
  return false;
}

namespace {
AST_MATCHER(BinaryOperator,
            hasBinaryOperatorAChildComparisonOperatorWithoutParen) {
  return isExprAComparisonOperator(Node.getLHS()) ||
         isExprAComparisonOperator(Node.getRHS());
}

AST_MATCHER(CXXOperatorCallExpr,
            hasCppOperatorAChildComparisonOperatorWithoutParen) {
  return llvm::any_of(Node.arguments(), isExprAComparisonOperator);
}

struct ChainedComparisonData {
  llvm::SmallString<256U> Name;
  llvm::SmallVector<const Expr *, 32U> Operands;

  explicit ChainedComparisonData(const Expr *Op) { extract(Op); }

private:
  void add(const Expr *Operand);
  void add(llvm::StringRef Opcode);
  void extract(const Expr *Op);
  void extract(const BinaryOperator *Op);
  void extract(const CXXOperatorCallExpr *Op);
};

} // namespace

void ChainedComparisonData::add(const Expr *Operand) {
  if (!Name.empty())
    Name += ' ';
  Name += 'v';
  Name += std::to_string(Operands.size());
  Operands.push_back(Operand);
}

void ChainedComparisonData::add(llvm::StringRef Opcode) {
  Name += ' ';
  Name += Opcode;
}

void ChainedComparisonData::extract(const BinaryOperator *Op) {
  const Expr *LHS = Op->getLHS()->IgnoreImplicit();
  if (isExprAComparisonOperator(LHS))
    extract(LHS);
  else
    add(LHS);

  add(Op->getOpcodeStr());

  const Expr *RHS = Op->getRHS()->IgnoreImplicit();
  if (isExprAComparisonOperator(RHS))
    extract(RHS);
  else
    add(RHS);
}

void ChainedComparisonData::extract(const CXXOperatorCallExpr *Op) {
  const Expr *FirstArg = Op->getArg(0U)->IgnoreImplicit();
  if (isExprAComparisonOperator(FirstArg))
    extract(FirstArg);
  else
    add(FirstArg);

  add(getOperatorSpelling(Op->getOperator()));

  const Expr *SecondArg = Op->getArg(1U)->IgnoreImplicit();
  if (isExprAComparisonOperator(SecondArg))
    extract(SecondArg);
  else
    add(SecondArg);
}

void ChainedComparisonData::extract(const Expr *Op) {
  if (!Op)
    return;

  if (const auto *BinaryOp = dyn_cast<BinaryOperator>(Op)) {
    extract(BinaryOp);
    return;
  }

  if (const auto *OverloadedOp = dyn_cast<CXXOperatorCallExpr>(Op)) {
    if (OverloadedOp->getNumArgs() == 2U)
      extract(OverloadedOp);
  }
}

void ChainedComparisonCheck::registerMatchers(MatchFinder *Finder) {
  const auto OperatorMatcher = expr(anyOf(
      binaryOperator(isComparisonOperator(),
                     hasBinaryOperatorAChildComparisonOperatorWithoutParen()),
      cxxOperatorCallExpr(
          isComparisonOperator(),
          hasCppOperatorAChildComparisonOperatorWithoutParen())));

  Finder->addMatcher(
      expr(OperatorMatcher, unless(hasParent(OperatorMatcher))).bind("op"),
      this);
}

void ChainedComparisonCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedOperator = Result.Nodes.getNodeAs<Expr>("op");

  ChainedComparisonData Data(MatchedOperator);
  if (Data.Operands.empty())
    return;

  diag(MatchedOperator->getBeginLoc(),
       "chained comparison '%0' may generate unintended results, use "
       "parentheses to specify order of evaluation or a logical operator to "
       "separate comparison expressions")
      << llvm::StringRef(Data.Name).trim() << MatchedOperator->getSourceRange();

  for (std::size_t Index = 0U; Index < Data.Operands.size(); ++Index) {
    diag(Data.Operands[Index]->getBeginLoc(), "operand 'v%0' is here",
         DiagnosticIDs::Note)
        << Index << Data.Operands[Index]->getSourceRange();
  }
}

} // namespace clang::tidy::bugprone
