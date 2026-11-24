//===-- JoinTest.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that pointers have the expected nullability state after a join
// operation. The tests check the analysis state directly (vs tests for
// emitted diagnostics).
//
//===----------------------------------------------------------------------===//

#include "CheckNullabilityState.h"
#include "clang/Basic/Specifiers.h"
#include "gtest/gtest.h"

namespace clang::dataflow::nullability {
namespace {

TEST(NullabilityJoinTest, JoinSameNullability) {
  // Joining when two branches have the same nullability.
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nonnull p1, int * _Nonnull p2, bool b) {
      int *target_p = p1;
      if (b) {
        target_p = p2;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nullable p1, int * _Nullable p2, bool b) {
      int *target_p = p1;
      if (b) {
        target_p = p2;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(bool b) {
      int *target_p = nullptr;
      if (b) {
        target_p = 0;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * p1, int *  p2, bool b) {
      int *target_p = p1;
      if (b) {
        target_p = p2;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Unspecified);
}

TEST(NullabilityJoinTest, JoinDifferentNullability) {
  // Joining when two branches have the different nullability.
  // Nonnull and nullable => nullable.
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nonnull nonnull_p, int * _Nullable nullable_p, bool b) {
      int *target_p = nonnull_p;
      if (b) {
        target_p = nullable_p;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  // Nonnull and nullptr
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nonnull nonnull_p, bool b) {
      int *target_p = nonnull_p;
      if (b) {
        target_p = nullptr;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  // Nullable and nullptr
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nullable nullable_p, bool b) {
      int *target_p = nullable_p;
      if (b) {
        target_p = nullptr;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  // Unspecified and nullable
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int *unknown_p, int * _Nullable nullable_p, bool b) {
      int *target_p = unknown_p;
      if (b) {
        target_p = nullable_p;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  // Unspecified and nullptr
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int *unknown_p, bool b) {
      int *target_p = unknown_p;
      if (b) {
        target_p = nullptr;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  // Unspecified and nonnull => unspecified.
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int *unknown_p, int * _Nonnull nonnull_p, bool b) {
      int *target_p = unknown_p;
      if (b) {
        target_p = nonnull_p;
      }
      (void)0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Unspecified);
}

} // namespace
} // namespace clang::dataflow::nullability
