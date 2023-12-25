//===--- ChainedComparisonCheck.cpp - clang-tidy --------------------------===//
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
#include <array>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

bool isExprAComparisonOperator(const Expr *E) {
  if (const auto *Op = dyn_cast_or_null<BinaryOperator>(E->IgnoreImplicit()))
    return Op->isComparisonOp();
  if (const auto *Op =
          dyn_cast_or_null<CXXOperatorCallExpr>(E->IgnoreImplicit()))
    return Op->isComparisonOp();
  return false;
}

AST_MATCHER(BinaryOperator,
            hasBinaryOperatorAChildComparisonOperatorWithoutParen) {
  return isExprAComparisonOperator(Node.getLHS()) ||
         isExprAComparisonOperator(Node.getRHS());
}

AST_MATCHER(CXXOperatorCallExpr,
            hasCppOperatorAChildComparisonOperatorWithoutParen) {
  return std::any_of(Node.arg_begin(), Node.arg_end(),
                     isExprAComparisonOperator);
}

constexpr std::array<llvm::StringRef, 26U> Letters = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};

struct ChainedComparisonData {
  llvm::SmallString<256U> Name;
  llvm::SmallVector<const Expr *, 26U> Operands;
  bool Full = false;

  void Add(const Expr *Operand) {
    if (Full)
      return;
    if (!Name.empty())
      Name += ' ';
    Name += Letters[Operands.size()];
    Operands.push_back(Operand);

    if (Operands.size() == Letters.size()) {
      Name += " ...";
      Full = true;
    }
  }

  void Add(llvm::StringRef Opcode) {
    if (Full)
      return;

    Name += ' ';
    Name += Opcode.str();
  }
};

} // namespace

static void extractData(const Expr *Op, ChainedComparisonData &Output);

inline bool extractData(const BinaryOperator *Op,
                        ChainedComparisonData &Output) {
  if (!Op)
    return false;

  if (isExprAComparisonOperator(Op->getLHS()))
    extractData(Op->getLHS()->IgnoreImplicit(), Output);
  else
    Output.Add(Op->getLHS()->IgnoreUnlessSpelledInSource());

  Output.Add(Op->getOpcodeStr());

  if (isExprAComparisonOperator(Op->getRHS()))
    extractData(Op->getRHS()->IgnoreImplicit(), Output);
  else
    Output.Add(Op->getRHS()->IgnoreUnlessSpelledInSource());
  return true;
}

inline bool extractData(const CXXOperatorCallExpr *Op,
                        ChainedComparisonData &Output) {
  if (!Op || Op->getNumArgs() != 2U)
    return false;

  const Expr *FirstArg = Op->getArg(0U)->IgnoreImplicit();
  if (isExprAComparisonOperator(FirstArg))
    extractData(FirstArg, Output);
  else
    Output.Add(FirstArg->IgnoreUnlessSpelledInSource());

  Output.Add(getOperatorSpelling(Op->getOperator()));

  const Expr *SecondArg = Op->getArg(1U)->IgnoreImplicit();
  if (isExprAComparisonOperator(SecondArg))
    extractData(SecondArg, Output);
  else
    Output.Add(SecondArg->IgnoreUnlessSpelledInSource());
  return true;
}

static void extractData(const Expr *OpExpr, ChainedComparisonData &Output) {
  OpExpr->dump();
  extractData(dyn_cast_or_null<BinaryOperator>(OpExpr), Output) ||
      extractData(dyn_cast_or_null<CXXOperatorCallExpr>(OpExpr), Output);
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

  ChainedComparisonData Data;
  extractData(MatchedOperator, Data);

  if (Data.Operands.empty())
    return;

  diag(MatchedOperator->getBeginLoc(),
       "chained comparison '%0' may generate unintended results, use "
       "parentheses to specify order of evaluation or a logical operator to "
       "separate comparison expressions")
      << llvm::StringRef(Data.Name).trim();

  for (std::size_t Index = 0U; Index < Data.Operands.size(); ++Index) {
    diag(Data.Operands[Index]->getBeginLoc(), "operand '%0' is here",
         DiagnosticIDs::Note)
        << Letters[Index];
  }
}

} // namespace clang::tidy::bugprone
