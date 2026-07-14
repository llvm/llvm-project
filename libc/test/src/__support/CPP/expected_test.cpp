//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Unit tests for LIBC_NAMESPACE::cpp::expected and unexpected.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/expected.h"
#include "src/__support/CPP/type_traits.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::expected;
using LIBC_NAMESPACE::cpp::unexpected;

TEST(LlvmLibcExpectedTest, Unexpected) {
  unexpected<int> u1(42);
  ASSERT_EQ(u1.error(), 42);
}

TEST(LlvmLibcExpectedTest, ValueConstruction) {
  expected<int, double> e(123);
  ASSERT_TRUE(e.has_value());
  ASSERT_TRUE(static_cast<bool>(e));
  ASSERT_EQ(e.value(), 123);
  ASSERT_EQ(*e, 123);
}

TEST(LlvmLibcExpectedTest, ErrorConstruction) {
  expected<int, int> e(unexpected(404));
  ASSERT_FALSE(e.has_value());
  ASSERT_FALSE(static_cast<bool>(e));
  ASSERT_EQ(e.error(), 404);
}

TEST(LlvmLibcExpectedTest, Mutation) {
  expected<int, int> e(10);
  ASSERT_EQ(e.value(), 10);
  e.value() = 20;
  ASSERT_EQ(e.value(), 20);
  *e = 30;
  ASSERT_EQ(*e, 30);

  expected<int, int> u(unexpected<int>(1));
  ASSERT_EQ(u.error(), 1);
  u.error() = 2;
  ASSERT_EQ(u.error(), 2);
}

TEST(LlvmLibcExpectedTest, ConstAccess) {
  const expected<int, int> CE(100);
  ASSERT_TRUE(CE.has_value());
  ASSERT_TRUE(static_cast<bool>(CE));
  ASSERT_EQ(CE.value(), 100);
  ASSERT_EQ(*CE, 100);

  const expected<int, int> CU(unexpected(500));
  ASSERT_FALSE(CU.has_value());
  ASSERT_FALSE(static_cast<bool>(CU));
  ASSERT_EQ(CU.error(), 500);
}

struct Foo {
  int x;
  int get_x() const { return x; }
  void set_x(int new_x) { x = new_x; }
};

TEST(LlvmLibcExpectedTest, ArrowOperator) {
  expected<Foo, int> e(Foo{42});
  ASSERT_TRUE(e.has_value());
  ASSERT_EQ(e->x, 42);
  ASSERT_EQ(e->get_x(), 42);
  e->set_x(99);
  ASSERT_EQ(e->x, 99);

  const expected<Foo, int> CE(Foo{123});
  ASSERT_EQ(CE->x, 123);
  ASSERT_EQ(CE->get_x(), 123);
}

constexpr bool test_constexpr_value() {
  expected<int, int> e(42);
  if (!e.has_value() || !static_cast<bool>(e))
    return false;
  if (e.value() != 42 || *e != 42)
    return false;
  return true;
}

static_assert(test_constexpr_value(), "expected constexpr value check failed");

constexpr bool test_constexpr_error() {
  expected<int, int> e(unexpected<int>(99));
  if (e.has_value() || static_cast<bool>(e))
    return false;
  if (e.error() != 99)
    return false;
  return true;
}

static_assert(test_constexpr_error(), "expected constexpr error check failed");

static_assert(!LIBC_NAMESPACE::cpp::is_convertible_v<expected<int, long>, bool>,
              "only explicit conversions allowed");
