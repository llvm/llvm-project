//===- ArenaTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Arena.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::dataflow {
namespace {

class ArenaTest : public ::testing::Test {
protected:
  Arena A;
};

TEST_F(ArenaTest, CreateAtomicBoolValueReturnsDistinctValues) {
  auto &X = A.create<AtomicBoolValue>();
  auto &Y = A.create<AtomicBoolValue>();
  EXPECT_NE(&X, &Y);
}

TEST_F(ArenaTest, CreateTopBoolValueReturnsDistinctValues) {
  auto &X = A.create<TopBoolValue>();
  auto &Y = A.create<TopBoolValue>();
  EXPECT_NE(&X, &Y);
}

TEST_F(ArenaTest, GetOrCreateConjunctionReturnsSameExprGivenSameArgs) {
  auto &X = A.create<AtomicBoolValue>();
  auto &XAndX = A.makeAnd(X, X);
  EXPECT_EQ(&XAndX, &X);
}

TEST_F(ArenaTest, GetOrCreateConjunctionReturnsSameExprOnSubsequentCalls) {
  auto &X = A.create<AtomicBoolValue>();
  auto &Y = A.create<AtomicBoolValue>();
  auto &XAndY1 = A.makeAnd(X, Y);
  auto &XAndY2 = A.makeAnd(X, Y);
  EXPECT_EQ(&XAndY1, &XAndY2);

  auto &YAndX = A.makeAnd(Y, X);
  EXPECT_EQ(&XAndY1, &YAndX);

  auto &Z = A.create<AtomicBoolValue>();
  auto &XAndZ = A.makeAnd(X, Z);
  EXPECT_NE(&XAndY1, &XAndZ);
}

TEST_F(ArenaTest, GetOrCreateDisjunctionReturnsSameExprGivenSameArgs) {
  auto &X = A.create<AtomicBoolValue>();
  auto &XOrX = A.makeOr(X, X);
  EXPECT_EQ(&XOrX, &X);
}

TEST_F(ArenaTest, GetOrCreateDisjunctionReturnsSameExprOnSubsequentCalls) {
  auto &X = A.create<AtomicBoolValue>();
  auto &Y = A.create<AtomicBoolValue>();
  auto &XOrY1 = A.makeOr(X, Y);
  auto &XOrY2 = A.makeOr(X, Y);
  EXPECT_EQ(&XOrY1, &XOrY2);

  auto &YOrX = A.makeOr(Y, X);
  EXPECT_EQ(&XOrY1, &YOrX);

  auto &Z = A.create<AtomicBoolValue>();
  auto &XOrZ = A.makeOr(X, Z);
  EXPECT_NE(&XOrY1, &XOrZ);
}

TEST_F(ArenaTest, GetOrCreateNegationReturnsSameExprOnSubsequentCalls) {
  auto &X = A.create<AtomicBoolValue>();
  auto &NotX1 = A.makeNot(X);
  auto &NotX2 = A.makeNot(X);
  EXPECT_EQ(&NotX1, &NotX2);

  auto &Y = A.create<AtomicBoolValue>();
  auto &NotY = A.makeNot(Y);
  EXPECT_NE(&NotX1, &NotY);
}

TEST_F(ArenaTest, GetOrCreateImplicationReturnsTrueGivenSameArgs) {
  auto &X = A.create<AtomicBoolValue>();
  auto &XImpliesX = A.makeImplies(X, X);
  EXPECT_EQ(&XImpliesX, &A.makeLiteral(true));
}

TEST_F(ArenaTest, GetOrCreateImplicationReturnsSameExprOnSubsequentCalls) {
  auto &X = A.create<AtomicBoolValue>();
  auto &Y = A.create<AtomicBoolValue>();
  auto &XImpliesY1 = A.makeImplies(X, Y);
  auto &XImpliesY2 = A.makeImplies(X, Y);
  EXPECT_EQ(&XImpliesY1, &XImpliesY2);

  auto &YImpliesX = A.makeImplies(Y, X);
  EXPECT_NE(&XImpliesY1, &YImpliesX);

  auto &Z = A.create<AtomicBoolValue>();
  auto &XImpliesZ = A.makeImplies(X, Z);
  EXPECT_NE(&XImpliesY1, &XImpliesZ);
}

TEST_F(ArenaTest, GetOrCreateIffReturnsTrueGivenSameArgs) {
  auto &X = A.create<AtomicBoolValue>();
  auto &XIffX = A.makeEquals(X, X);
  EXPECT_EQ(&XIffX, &A.makeLiteral(true));
}

TEST_F(ArenaTest, GetOrCreateIffReturnsSameExprOnSubsequentCalls) {
  auto &X = A.create<AtomicBoolValue>();
  auto &Y = A.create<AtomicBoolValue>();
  auto &XIffY1 = A.makeEquals(X, Y);
  auto &XIffY2 = A.makeEquals(X, Y);
  EXPECT_EQ(&XIffY1, &XIffY2);

  auto &YIffX = A.makeEquals(Y, X);
  EXPECT_EQ(&XIffY1, &YIffX);

  auto &Z = A.create<AtomicBoolValue>();
  auto &XIffZ = A.makeEquals(X, Z);
  EXPECT_NE(&XIffY1, &XIffZ);
}

} // namespace
} // namespace clang::dataflow
