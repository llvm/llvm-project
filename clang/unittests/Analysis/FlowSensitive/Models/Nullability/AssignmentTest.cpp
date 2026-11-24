//===-- AssignmentTest.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that pointers have the expected nullability state propagated across
// initialization and assignments. The tests check the analysis state directly
// (vs tests for emitted diagnostics).
//
//===----------------------------------------------------------------------===//

#include "CheckNullabilityState.h"
#include "clang/Basic/Specifiers.h"
#include "gtest/gtest.h"

namespace clang::dataflow::nullability {
namespace {

TEST(NullabilityAssignmentTest, AddrOf) {
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target() {
      int local = 0;
      int *target_p = &local;
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);
}

TEST(NullabilityAssignmentTest, NullPointer) {
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target() {
      int *target_p = nullptr;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target() {
      int *target_p = 0;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
  namespace std {
    using nullptr_t = decltype(nullptr);
}
    void target(std::nullptr_t null_val) {
      int *target_p = null_val;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);
}

TEST(NullabilityAssignmentTest, AssignFromAnnotatedPointer) {
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nonnull nonnull_p) {
      int *target_p = nonnull_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nullable nullable_p) {
      int *target_p = nullable_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int *unknown_p) {
      int *target_p = unknown_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Unspecified);
}

TEST(NullabilityAssignmentTest, AssignmentOverwrite) {
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nonnull nonnull_p) {
      int *target_p = nullptr;
      target_p = nonnull_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nonnull nonnull_p) {
      int *target_p = nonnull_p;
      target_p = nullptr;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * unspecified_p) {
      int *target_p = nullptr;
      target_p = unspecified_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Unspecified);
}

TEST(NullabilityAssignmentTest, AssignFromAnnotatedTypeAlias) {
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    using NonnullPtr = int *_Nonnull;
    void target(NonnullPtr nonnull_p) {
      int *target_p = nonnull_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    typedef int* _Nonnull NonnullPtr;
    void target(NonnullPtr nonnull_p) {
      int *target_p = nonnull_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    using IntPtr = int *;
    void target(_Nullable IntPtr nullable_p) {
      int *target_p = nullable_p;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);
}

TEST(NullabilityAssignmentTest, AssignFromInnerPointerType) {
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nonnull * _Nonnull pp) {
      int *target_p = *pp;
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int * _Nullable * _Nonnull pp) {
      int *target_p = *pp;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Nullable);

  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int** _Nonnull pp) {
      int *target_p = *pp;
      // [[check]]
    }
  )cc"),
            NullabilityKind::Unspecified);
}

TEST(NullabilityAssignmentTest, BitCast) {
  EXPECT_EQ(evaluateNullabilityAtCheck(R"cc(
    void target(int local) {
      void *v = &local;
      int *target_p = static_cast<int *>(v);
      // [[check]]
    }
  )cc"),
            NullabilityKind::NonNull);
}

} // namespace
} // namespace clang::dataflow::nullability
