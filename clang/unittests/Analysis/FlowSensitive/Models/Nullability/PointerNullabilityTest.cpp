//===-- PointerNullabilityTest.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Low-level unit tests for the pointer value null state (IsNull, FromNullable).
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullability.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/Specifiers.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang::dataflow::nullability {
namespace {

class NullabilityPropertiesTest : public ::testing::Test {
protected:
  PointerValue &makePointer(NullabilityKind N) {
    auto &P = Env.create<PointerValue>(DACtx.createStorageLocation(QualType()));
    initPointerNullState(P, DACtx, N);
    return P;
  }

  PointerValue &makeNullPointer() {
    auto &P = Env.create<PointerValue>(DACtx.createStorageLocation(QualType()));
    initNullPointer(P, DACtx);
    return P;
  }

  DataflowAnalysisContext DACtx =
      DataflowAnalysisContext(std::make_unique<WatchedLiteralsSolver>());
  Environment Env = Environment(DACtx);
  Arena &A = DACtx.arena();
};

using IsNullableTest = NullabilityPropertiesTest;

TEST_F(IsNullableTest, NullPtr) {
  EXPECT_TRUE(isNullable(makeNullPointer(), Env));
}

TEST_F(IsNullableTest, NullableButNotNull) {
  auto &NullableButNotNull = makePointer(NullabilityKind::Nullable);
  EXPECT_TRUE(isNullable(NullableButNotNull, Env));
  auto *IsNull = getPointerNullState(NullableButNotNull).IsNull;
  ASSERT_NE(IsNull, nullptr);
  Env.assume(A.makeNot(*IsNull));

  EXPECT_FALSE(isNullable(NullableButNotNull, Env));
}

TEST_F(IsNullableTest, NullableAndNull) {
  auto &NullableAndNull = makePointer(NullabilityKind::Nullable);
  auto *IsNull = getPointerNullState(NullableAndNull).IsNull;
  ASSERT_NE(IsNull, nullptr);
  Env.assume(*IsNull);

  EXPECT_TRUE(isNullable(NullableAndNull, Env));
}

TEST_F(IsNullableTest, NonnullAndNotNull) {
  auto &NonnullAndNotNull = makePointer(NullabilityKind::NonNull);
  EXPECT_FALSE(isNullable(NonnullAndNotNull, Env));

  auto *IsNull = getPointerNullState(NonnullAndNotNull).IsNull;
  ASSERT_NE(IsNull, nullptr);

  // `IsNull` should not just be provably false but an actual false literal.
  ASSERT_EQ(IsNull, &A.makeLiteral(false));

  // Assuming the pointer is non-null is a no-op, but make sure it doesn't
  // change the result of `isNullable()`.
  Env.assume(A.makeNot(*IsNull));
  EXPECT_FALSE(isNullable(NonnullAndNotNull, Env));
}

TEST_F(IsNullableTest, NonnullAndNull) {
  // If a pointer comes from a non-null source but is dynamically discovered to
  // be definitely null, we don't consider it nullable, because we're in an
  // environment with false flow conditions.
  auto &NonnullAndNull = makePointer(NullabilityKind::NonNull);
  auto *IsNull = getPointerNullState(NonnullAndNull).IsNull;
  ASSERT_NE(IsNull, nullptr);
  Env.assume(*IsNull);

  EXPECT_FALSE(isNullable(NonnullAndNull, Env));
}

TEST_F(IsNullableTest, UnknownAndNull) {
  // If a pointer comes from an unknown source but is dynamically discovered to
  // be definitely null, we consider it nullable.
  auto &UnknownAndNull = makePointer(NullabilityKind::Unspecified);
  auto *IsNull = getPointerNullState(UnknownAndNull).IsNull;
  ASSERT_NE(IsNull, nullptr);
  Env.assume(*IsNull);

  EXPECT_TRUE(isNullable(UnknownAndNull, Env));
}

TEST_F(NullabilityPropertiesTest, InitNullabilityPropertiesWithFormulas) {
  auto &P = Env.create<PointerValue>(DACtx.createStorageLocation(QualType()));

  Arena &A = DACtx.arena();
  const Formula &FromNullable = A.makeAtomRef(A.makeAtom());
  const Formula &IsNull = A.makeAtomRef(A.makeAtom());

  initPointerNullState(P, DACtx, PointerNullState{&FromNullable, &IsNull});
  ASSERT_EQ(getPointerNullState(P).FromNullable, &FromNullable);
  ASSERT_EQ(getPointerNullState(P).IsNull, &IsNull);
}

TEST_F(NullabilityPropertiesTest, InitNullabilityPropertiesWithTop) {
  auto &P = Env.create<PointerValue>(DACtx.createStorageLocation(QualType()));

  initPointerNullState(P, DACtx, PointerNullState{nullptr, nullptr});
  ASSERT_EQ(getPointerNullState(P).FromNullable, nullptr);
  ASSERT_EQ(getPointerNullState(P).IsNull, nullptr);
}

} // namespace
} // namespace clang::dataflow::nullability
