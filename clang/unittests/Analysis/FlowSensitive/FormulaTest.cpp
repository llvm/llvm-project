//===- unittests/Analysis/FlowSensitive/FormulaTest.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Formula.h"
#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/FormulaSerialization.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using namespace clang;
using namespace dataflow;

using ::llvm::HasValue;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

class SerializeFormulaTest : public ::testing::Test {
protected:
  Arena A;
  std::string Out;
  llvm::raw_string_ostream OS{Out};

  const Formula &A1 = A.makeAtomRef(A.makeAtom());
  const Formula &A2 = A.makeAtomRef(A.makeAtom());
};

TEST_F(SerializeFormulaTest, Atom) {
  serializeFormula(A1, OS);
  EXPECT_EQ(Out, "V0");
  Out = "";

  serializeFormula(A2, OS);
  EXPECT_EQ(Out, "V1");
}

TEST_F(SerializeFormulaTest, LiteralTrue) {
  serializeFormula(A.makeLiteral(true), OS);
  EXPECT_EQ(Out, "T");
}

TEST_F(SerializeFormulaTest, LiteralFalse) {
  serializeFormula(A.makeLiteral(false), OS);
  EXPECT_EQ(Out, "F");
}

TEST_F(SerializeFormulaTest, Not) {
  serializeFormula(A.makeNot(A1), OS);
  EXPECT_EQ(Out, "!V0");
}

TEST_F(SerializeFormulaTest, Or) {
  serializeFormula(A.makeOr(A1, A2), OS);
  EXPECT_EQ(Out, "|V0V1");
}

TEST_F(SerializeFormulaTest, And) {
  serializeFormula(A.makeAnd(A1, A2), OS);
  EXPECT_EQ(Out, "&V0V1");
}

TEST_F(SerializeFormulaTest, Implies) {
  serializeFormula(A.makeImplies(A1, A2), OS);
  EXPECT_EQ(Out, ">V0V1");
}

TEST_F(SerializeFormulaTest, Equal) {
  serializeFormula(A.makeEquals(A1, A2), OS);
  EXPECT_EQ(Out, "=V0V1");
}

TEST_F(SerializeFormulaTest, NestedBinaryUnary) {
  serializeFormula(A.makeEquals(A.makeOr(A1, A2), A2), OS);
  EXPECT_EQ(Out, "=|V0V1V1");
}

TEST_F(SerializeFormulaTest, NestedBinaryBinary) {
  serializeFormula(A.makeEquals(A.makeOr(A1, A2), A.makeAnd(A1, A2)), OS);
  EXPECT_EQ(Out, "=|V0V1&V0V1");
}

class ParseFormulaTest : public ::testing::Test {
protected:
  void SetUp() override {
    AtomMap[0] = Atom1;
    AtomMap[1] = Atom2;
  }

  Arena A;
  std::string Out;
  llvm::raw_string_ostream OS{Out};

  Atom Atom1 = A.makeAtom();
  Atom Atom2 = A.makeAtom();
  const Formula &A1 = A.makeAtomRef(Atom1);
  const Formula &A2 = A.makeAtomRef(Atom2);
  llvm::DenseMap<unsigned, Atom> AtomMap;
};

TEST_F(ParseFormulaTest, Atom) {
  EXPECT_THAT_EXPECTED(parseFormula("V0", A, AtomMap), HasValue(&A1));
  EXPECT_THAT_EXPECTED(parseFormula("V1", A, AtomMap), HasValue(&A2));
}

TEST_F(ParseFormulaTest, LiteralTrue) {
  EXPECT_THAT_EXPECTED(parseFormula("T", A, AtomMap),
                       HasValue(&A.makeLiteral(true)));
}

TEST_F(ParseFormulaTest, LiteralFalse) {
  EXPECT_THAT_EXPECTED(parseFormula("F", A, AtomMap),
                       HasValue(&A.makeLiteral(false)));
}

TEST_F(ParseFormulaTest, Not) {
  EXPECT_THAT_EXPECTED(parseFormula("!V0", A, AtomMap),
                       HasValue(&A.makeNot(A1)));
}

TEST_F(ParseFormulaTest, Or) {
  EXPECT_THAT_EXPECTED(parseFormula("|V0V1", A, AtomMap),
                       HasValue(&A.makeOr(A1, A2)));
}

TEST_F(ParseFormulaTest, And) {
  EXPECT_THAT_EXPECTED(parseFormula("&V0V1", A, AtomMap),
                       HasValue(&A.makeAnd(A1, A2)));
}

TEST_F(ParseFormulaTest, Implies) {
  EXPECT_THAT_EXPECTED(parseFormula(">V0V1", A, AtomMap),
                       HasValue(&A.makeImplies(A1, A2)));
}

TEST_F(ParseFormulaTest, Equal) {
  EXPECT_THAT_EXPECTED(parseFormula("=V0V1", A, AtomMap),
                       HasValue(&A.makeEquals(A1, A2)));
}

TEST_F(ParseFormulaTest, NestedBinaryUnary) {
  EXPECT_THAT_EXPECTED(parseFormula("=|V0V1V1", A, AtomMap),
                       HasValue(&A.makeEquals(A.makeOr(A1, A2), A2)));
}

TEST_F(ParseFormulaTest, NestedBinaryBinary) {
  EXPECT_THAT_EXPECTED(
      parseFormula("=|V0V1&V0V1", A, AtomMap),
      HasValue(&A.makeEquals(A.makeOr(A1, A2), A.makeAnd(A1, A2))));
}

// Verifies that parsing generates fresh atoms, if they are not already in the
// map.
TEST_F(ParseFormulaTest, GeneratesAtoms) {
  llvm::DenseMap<unsigned, Atom> FreshAtomMap;
  ASSERT_THAT_EXPECTED(parseFormula("=V0V1", A, FreshAtomMap),
                       llvm::Succeeded());
  // The map contains two, unique elements.
  ASSERT_EQ(FreshAtomMap.size(), 2U);
  EXPECT_NE(FreshAtomMap[0], FreshAtomMap[1]);
}

TEST_F(ParseFormulaTest, BadFormulaFails) {
  EXPECT_THAT_EXPECTED(parseFormula("Hello", A, AtomMap), llvm::Failed());
}

TEST_F(ParseFormulaTest, FormulaWithSuffixFails) {
  EXPECT_THAT_EXPECTED(parseFormula("=V0V1Hello", A, AtomMap), llvm::Failed());
}

} // namespace
