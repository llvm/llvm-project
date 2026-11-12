//===- unittests/Basic/SanitizersTest.cpp - Test Sanitizers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Sanitizers.h"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace clang;

using testing::Contains;
using testing::Not;

TEST(SanitizersTest, serializeSanitizers) {
  SanitizerSet Set;
  Set.set(parseSanitizerValue("memory", false), true);
  Set.set(parseSanitizerValue("nullability-arg", false), true);

  SmallVector<StringRef, 4> Serialized;
  serializeSanitizerSet(Set, Serialized);

  ASSERT_EQ(Serialized.size(), 2u);
  ASSERT_THAT(Serialized, Contains("memory"));
  ASSERT_THAT(Serialized, Contains("nullability-arg"));
}

TEST(SanitizersTest, serializeSanitizersIndividual) {
  SanitizerSet Set;
  Set.set(parseSanitizerValue("memory", false), true);
  Set.set(parseSanitizerValue("nullability-arg", false), true);
  Set.set(parseSanitizerValue("nullability-assign", false), true);
  Set.set(parseSanitizerValue("nullability-return", false), true);

  SmallVector<StringRef, 4> Serialized;
  serializeSanitizerSet(Set, Serialized);

  ASSERT_EQ(Serialized.size(), 4u);
  ASSERT_THAT(Serialized, Contains("memory"));
  ASSERT_THAT(Serialized, Contains("nullability-arg"));
  ASSERT_THAT(Serialized, Contains("nullability-assign"));
  ASSERT_THAT(Serialized, Contains("nullability-return"));
  // Individual sanitizers don't get squashed into a single group.
  ASSERT_THAT(Serialized, Not(Contains("nullability")));
}
