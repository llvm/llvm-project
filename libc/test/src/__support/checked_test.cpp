//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for Checked.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/checked.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Checked;
using LIBC_NAMESPACE::cpp::is_constructible_v;
using LIBC_NAMESPACE::cpp::is_convertible_v;
using LIBC_NAMESPACE::cpp::move;

// Verify compile-time properties:
// - Implicitly constructible from bool, but not from other types.
// - Movable, but not copyable.
// - Explicit conversion to bool is only allowed on non-const lvalues/rvalues.
static_assert(!is_constructible_v<Checked>);
static_assert(is_convertible_v<bool, Checked>);
static_assert(!is_constructible_v<Checked, int>);
static_assert(!is_constructible_v<Checked, int *>);
static_assert(is_constructible_v<Checked, Checked &&>);
static_assert(!is_constructible_v<Checked, const Checked &>);
static_assert(!is_convertible_v<Checked, bool>);
static_assert(is_constructible_v<bool, Checked &>);
static_assert(is_constructible_v<bool, Checked &&>);
static_assert(!is_constructible_v<bool, const Checked &>);

static Checked succeed() { return true; }
static Checked fail() { return false; }

TEST(LlvmLibcCheckedTest, Value) {
  EXPECT_TRUE(static_cast<bool>(succeed()));
  EXPECT_FALSE(static_cast<bool>(fail()));
}

TEST(LlvmLibcCheckedTest, Move) {
  Checked c1 = succeed();
  // Moving disarms `c1`, and `c2` needs to be checked instead.
  Checked c2(move(c1));
  EXPECT_TRUE(static_cast<bool>(c2));

  Checked c3 = fail();
  EXPECT_FALSE(static_cast<bool>(c3));
  c3 = move(c2);
  EXPECT_TRUE(static_cast<bool>(c3));
}

#if !defined(NDEBUG) && LIBC_TEST_SUBPROCESS_TESTS

#if defined(LIBC_FULL_BUILD) && !defined(LIBC_COPT_USE_C_ASSERT)
#define EXPECT_ASSERT_FAILS(stmt) EXPECT_EXITS(stmt, 255)
#else
#define EXPECT_ASSERT_FAILS(stmt) EXPECT_DEATH(stmt, WITH_SIGNAL(-1))
#endif

TEST(LlvmLibcCheckedDeathTest, UncheckedDestructionDies) {
  EXPECT_ASSERT_FAILS([] {
    Checked c = succeed();
    (void)c;
  });
}

TEST(LlvmLibcCheckedDeathTest, MoveFromCheckedRequiresRecheckOrDies) {
  EXPECT_ASSERT_FAILS([] {
    Checked c1 = succeed();
    (void)static_cast<bool>(c1);
    Checked c2(move(c1));
  });
}

TEST(LlvmLibcCheckedDeathTest, OverwritingUncheckedViaMoveAssignmentDies) {
  EXPECT_ASSERT_FAILS([] {
    Checked c1 = succeed();
    Checked c2 = succeed();
    (void)static_cast<bool>(c2);
    c1 = move(c2);
    (void)static_cast<bool>(c1);
  });
}

#endif // !defined(NDEBUG) && LIBC_TEST_SUBPROCESS_TESTS
