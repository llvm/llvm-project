//===- unittests/Analysis/FlowSensitive/ValueTest.cpp ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;

TEST(ValueTest, EquivalenceReflexive) {
  IntegerValue V;
  EXPECT_TRUE(areEquivalentValues(V, V));
}

TEST(ValueTest, DifferentIntegerValuesNotEquivalent) {
  IntegerValue V1;
  IntegerValue V2;
  EXPECT_FALSE(areEquivalentValues(V1, V2));
}

TEST(ValueTest, AliasedReferencesEquivalent) {
  auto L = ScalarStorageLocation(QualType());
  ReferenceValue V1(L);
  ReferenceValue V2(L);
  EXPECT_TRUE(areEquivalentValues(V1, V2));
  EXPECT_TRUE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, AliasedPointersEquivalent) {
  auto L = ScalarStorageLocation(QualType());
  PointerValue V1(L);
  PointerValue V2(L);
  EXPECT_TRUE(areEquivalentValues(V1, V2));
  EXPECT_TRUE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, TopsEquivalent) {
  Arena A;
  TopBoolValue V1(A.makeAtomRef(Atom(0)));
  TopBoolValue V2(A.makeAtomRef(Atom(1)));
  EXPECT_TRUE(areEquivalentValues(V1, V2));
  EXPECT_TRUE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, EquivalentValuesWithDifferentPropsEquivalent) {
  Arena A;
  TopBoolValue Prop1(A.makeAtomRef(Atom(0)));
  TopBoolValue Prop2(A.makeAtomRef(Atom(1)));
  TopBoolValue V1(A.makeAtomRef(Atom(2)));
  TopBoolValue V2(A.makeAtomRef(Atom(3)));
  V1.setProperty("foo", Prop1);
  V2.setProperty("bar", Prop2);
  EXPECT_TRUE(areEquivalentValues(V1, V2));
  EXPECT_TRUE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, DifferentKindsNotEquivalent) {
  Arena A;
  auto L = ScalarStorageLocation(QualType());
  ReferenceValue V1(L);
  TopBoolValue V2(A.makeAtomRef(Atom(0)));
  EXPECT_FALSE(areEquivalentValues(V1, V2));
  EXPECT_FALSE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, NotAliasedReferencesNotEquivalent) {
  auto L1 = ScalarStorageLocation(QualType());
  ReferenceValue V1(L1);
  auto L2 = ScalarStorageLocation(QualType());
  ReferenceValue V2(L2);
  EXPECT_FALSE(areEquivalentValues(V1, V2));
  EXPECT_FALSE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, NotAliasedPointersNotEquivalent) {
  auto L1 = ScalarStorageLocation(QualType());
  PointerValue V1(L1);
  auto L2 = ScalarStorageLocation(QualType());
  PointerValue V2(L2);
  EXPECT_FALSE(areEquivalentValues(V1, V2));
  EXPECT_FALSE(areEquivalentValues(V2, V1));
}

} // namespace
