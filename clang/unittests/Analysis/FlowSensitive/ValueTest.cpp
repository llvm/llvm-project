//===- unittests/Analysis/FlowSensitive/ValueTest.cpp ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;

TEST(ValueTest, EquivalenceReflexive) {
  StructValue V;
  EXPECT_TRUE(areEquivalentValues(V, V));
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
  TopBoolValue V1;
  TopBoolValue V2;
  EXPECT_TRUE(areEquivalentValues(V1, V2));
  EXPECT_TRUE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, EquivalentValuesWithDifferentPropsEquivalent) {
  TopBoolValue Prop1;
  TopBoolValue Prop2;
  TopBoolValue V1;
  TopBoolValue V2;
  V1.setProperty("foo", Prop1);
  V2.setProperty("bar", Prop2);
  EXPECT_TRUE(areEquivalentValues(V1, V2));
  EXPECT_TRUE(areEquivalentValues(V2, V1));
}

TEST(ValueTest, DifferentKindsNotEquivalent) {
  auto L = ScalarStorageLocation(QualType());
  ReferenceValue V1(L);
  TopBoolValue V2;
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
