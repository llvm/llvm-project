//===- ArenaTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Arena.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::dataflow {
namespace {
using llvm::HasValue;
using testing::Ref;

class ArenaTest : public ::testing::Test {
protected:
  Arena A;
};

TEST_F(ArenaTest, CreateAtomicBoolValueReturnsDistinctValues) {
  auto &X = A.makeAtomValue();
  auto &Y = A.makeAtomValue();
  EXPECT_NE(&X, &Y);
}

TEST_F(ArenaTest, CreateTopBoolValueReturnsDistinctValues) {
  auto &X = A.makeTopValue();
  auto &Y = A.makeTopValue();
  EXPECT_NE(&X, &Y);
}

TEST_F(ArenaTest, GetOrCreateConjunctionReturnsSameExprOnSubsequentCalls) {
  auto &X = A.makeAtomRef(A.makeAtom());
  auto &Y = A.makeAtomRef(A.makeAtom());
  auto &XAndY1 = A.makeAnd(X, Y);
  auto &XAndY2 = A.makeAnd(X, Y);
  EXPECT_EQ(&XAndY1, &XAndY2);

  auto &YAndX = A.makeAnd(Y, X);
  EXPECT_EQ(&XAndY1, &YAndX);

  auto &Z = A.makeAtomRef(A.makeAtom());
  auto &XAndZ = A.makeAnd(X, Z);
  EXPECT_NE(&XAndY1, &XAndZ);
}

TEST_F(ArenaTest, GetOrCreateDisjunctionReturnsSameExprOnSubsequentCalls) {
  auto &X = A.makeAtomRef(A.makeAtom());
  auto &Y = A.makeAtomRef(A.makeAtom());
  auto &XOrY1 = A.makeOr(X, Y);
  auto &XOrY2 = A.makeOr(X, Y);
  EXPECT_EQ(&XOrY1, &XOrY2);

  auto &YOrX = A.makeOr(Y, X);
  EXPECT_EQ(&XOrY1, &YOrX);

  auto &Z = A.makeAtomRef(A.makeAtom());
  auto &XOrZ = A.makeOr(X, Z);
  EXPECT_NE(&XOrY1, &XOrZ);
}

TEST_F(ArenaTest, GetOrCreateNegationReturnsSameExprOnSubsequentCalls) {
  auto &X = A.makeAtomRef(A.makeAtom());
  auto &NotX1 = A.makeNot(X);
  auto &NotX2 = A.makeNot(X);
  EXPECT_EQ(&NotX1, &NotX2);
  auto &Y = A.makeAtomRef(A.makeAtom());
  auto &NotY = A.makeNot(Y);
  EXPECT_NE(&NotX1, &NotY);
}

TEST_F(ArenaTest, GetOrCreateImplicationReturnsSameExprOnSubsequentCalls) {
  auto &X = A.makeAtomRef(A.makeAtom());
  auto &Y = A.makeAtomRef(A.makeAtom());
  auto &XImpliesY1 = A.makeImplies(X, Y);
  auto &XImpliesY2 = A.makeImplies(X, Y);
  EXPECT_EQ(&XImpliesY1, &XImpliesY2);

  auto &YImpliesX = A.makeImplies(Y, X);
  EXPECT_NE(&XImpliesY1, &YImpliesX);

  auto &Z = A.makeAtomRef(A.makeAtom());
  auto &XImpliesZ = A.makeImplies(X, Z);
  EXPECT_NE(&XImpliesY1, &XImpliesZ);
}

TEST_F(ArenaTest, GetOrCreateIffReturnsSameExprOnSubsequentCalls) {
  auto &X = A.makeAtomRef(A.makeAtom());
  auto &Y = A.makeAtomRef(A.makeAtom());
  auto &XIffY1 = A.makeEquals(X, Y);
  auto &XIffY2 = A.makeEquals(X, Y);
  EXPECT_EQ(&XIffY1, &XIffY2);

  auto &YIffX = A.makeEquals(Y, X);
  EXPECT_EQ(&XIffY1, &YIffX);

  auto &Z = A.makeAtomRef(A.makeAtom());
  auto &XIffZ = A.makeEquals(X, Z);
  EXPECT_NE(&XIffY1, &XIffZ);
}

TEST_F(ArenaTest, Interning) {
  Atom X = A.makeAtom();
  Atom Y = A.makeAtom();
  const Formula &F1 = A.makeAnd(A.makeAtomRef(X), A.makeAtomRef(Y));
  const Formula &F2 = A.makeAnd(A.makeAtomRef(Y), A.makeAtomRef(X));
  EXPECT_EQ(&F1, &F2);
  BoolValue &B1 = A.makeBoolValue(F1);
  BoolValue &B2 = A.makeBoolValue(F2);
  EXPECT_EQ(&B1, &B2);
  EXPECT_EQ(&B1.formula(), &F1);
}

TEST_F(ArenaTest, ParseFormula) {
  Atom V5{5};
  Atom V6{6};
  EXPECT_THAT_EXPECTED(A.parseFormula("V5"), HasValue(Ref(A.makeAtomRef(V5))));
  EXPECT_THAT_EXPECTED(A.parseFormula("true"),
                       HasValue(Ref(A.makeLiteral(true))));
  EXPECT_THAT_EXPECTED(A.parseFormula("!V5"),
                       HasValue(Ref(A.makeNot(A.makeAtomRef(V5)))));

  EXPECT_THAT_EXPECTED(
      A.parseFormula("(V5 = V6)"),
      HasValue(Ref(A.makeEquals(A.makeAtomRef(V5), A.makeAtomRef(V6)))));
  EXPECT_THAT_EXPECTED(
      A.parseFormula("(V5 => V6)"),
      HasValue(Ref(A.makeImplies(A.makeAtomRef(V5), A.makeAtomRef(V6)))));
  EXPECT_THAT_EXPECTED(
      A.parseFormula("(V5 & V6)"),
      HasValue(Ref(A.makeAnd(A.makeAtomRef(V5), A.makeAtomRef(V6)))));
  EXPECT_THAT_EXPECTED(
      A.parseFormula("(V5 | V6)"),
      HasValue(Ref(A.makeOr(A.makeAtomRef(V5), A.makeAtomRef(V6)))));

  EXPECT_THAT_EXPECTED(
      A.parseFormula("((V5 & (V6 & !false)) => ((V5 | V6) | false))"),
      HasValue(Ref(
          A.makeImplies(A.makeAnd(A.makeAtomRef(V5),
                                  A.makeAnd(A.makeAtomRef(V6),
                                            A.makeNot(A.makeLiteral(false)))),
                        A.makeOr(A.makeOr(A.makeAtomRef(V5), A.makeAtomRef(V6)),
                                 A.makeLiteral(false))))));

  EXPECT_THAT_EXPECTED(
      A.parseFormula("(V0 => error)"), llvm::FailedWithMessage(R"(bad formula at offset 7
(V0 => error)
       ^)"));
  EXPECT_THAT_EXPECTED(
      A.parseFormula("V1 V2"), llvm::FailedWithMessage(R"(bad formula at offset 3
V1 V2
   ^)"));
}

TEST_F(ArenaTest, IdentitySimplification) {
  auto &X = A.makeAtomRef(A.makeAtom());

  EXPECT_EQ(&X, &A.makeAnd(X, X));
  EXPECT_EQ(&X, &A.makeOr(X, X));
  EXPECT_EQ(&A.makeLiteral(true), &A.makeImplies(X, X));
  EXPECT_EQ(&A.makeLiteral(true), &A.makeEquals(X, X));
  EXPECT_EQ(&X, &A.makeNot(A.makeNot(X)));
}

TEST_F(ArenaTest, LiteralSimplification) {
  auto &X = A.makeAtomRef(A.makeAtom());

  EXPECT_EQ(&X, &A.makeAnd(X, A.makeLiteral(true)));
  EXPECT_EQ(&A.makeLiteral(false), &A.makeAnd(X, A.makeLiteral(false)));

  EXPECT_EQ(&A.makeLiteral(true), &A.makeOr(X, A.makeLiteral(true)));
  EXPECT_EQ(&X, &A.makeOr(X, A.makeLiteral(false)));

  EXPECT_EQ(&A.makeLiteral(true), &A.makeImplies(X, A.makeLiteral(true)));
  EXPECT_EQ(&A.makeNot(X), &A.makeImplies(X, A.makeLiteral(false)));
  EXPECT_EQ(&X, &A.makeImplies(A.makeLiteral(true), X));
  EXPECT_EQ(&A.makeLiteral(true), &A.makeImplies(A.makeLiteral(false), X));

  EXPECT_EQ(&X, &A.makeEquals(X, A.makeLiteral(true)));
  EXPECT_EQ(&A.makeNot(X), &A.makeEquals(X, A.makeLiteral(false)));

  EXPECT_EQ(&A.makeLiteral(false), &A.makeNot(A.makeLiteral(true)));
  EXPECT_EQ(&A.makeLiteral(true), &A.makeNot(A.makeLiteral(false)));
}

} // namespace
} // namespace clang::dataflow
