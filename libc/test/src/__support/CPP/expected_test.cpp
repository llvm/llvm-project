//===-- Unittests for Expected --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/expected.h"
#include "test/UnitTest/Test.h"

using __llvm_libc::cpp::expected;
using __llvm_libc::cpp::unexpected;

enum class error { _ = 0, B };
enum class fatal_error { _ = 0, D };

auto make_error() { return unexpected<error>{error::B}; }

TEST(LlvmLibcExpectedTest, DefaultCtor) {
  {
    expected<int, error> init;
    EXPECT_TRUE(init.has_value());
    EXPECT_TRUE((bool)init);
    EXPECT_EQ(init.value(), int{});
    EXPECT_EQ(*init, int{});
  }
  {
    expected<void, error> init;
    EXPECT_TRUE(init.has_value());
    EXPECT_TRUE((bool)init);
  }
}

TEST(LlvmLibcExpectedTest, ValueCtor) {
  expected<int, error> init(1);
  EXPECT_TRUE(init.has_value());
  EXPECT_TRUE((bool)init);
  EXPECT_EQ(init.value(), 1);
  EXPECT_EQ(*init, 1);
}

TEST(LlvmLibcExpectedTest, ErrorCtor) {
  {
    expected<int, error> init = make_error();
    EXPECT_FALSE(init.has_value());
    EXPECT_FALSE((bool)init);
    EXPECT_EQ(init.error(), error::B);
  }
  {
    expected<void, error> init = make_error();
    EXPECT_FALSE(init.has_value());
    EXPECT_FALSE((bool)init);
    EXPECT_EQ(init.error(), error::B);
  }
}

// value_or
TEST(LlvmLibcExpectedTest, ValueOr_ReuseValue) {
  expected<int, error> init(1);
  EXPECT_EQ(init.value_or(2), int(1));
}

TEST(LlvmLibcExpectedTest, ValueOr_UseProvided) {
  expected<int, error> init = make_error();
  EXPECT_EQ(init.value_or(2), int(2));
}

// transform

TEST(LlvmLibcExpectedTest, Transform_HasValue) {
  { // int to int
    expected<int, error> init(1);
    expected<int, error> expected_next(2);
    expected<int, error> actual_next = init.transform([](int) { return 2; });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to int
    expected<void, error> init;
    expected<int, error> expected_next(2);
    expected<int, error> actual_next = init.transform([]() { return 2; });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // int to void
    expected<int, error> init(1);
    expected<void, error> expected_next;
    expected<void, error> actual_next = init.transform([](int) {});
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to void
    expected<void, error> init;
    expected<void, error> actual_next = init.transform([]() {});
    EXPECT_TRUE(actual_next == init);
  }
}

TEST(LlvmLibcExpectedTest, Transform_HasError) {
  { // int to int
    expected<int, error> init = make_error();
    expected<int, error> expected_next = make_error();
    expected<int, error> actual_next = init.transform([](int) { return 2; });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to int
    expected<void, error> init = make_error();
    expected<int, error> expected_next = make_error();
    expected<int, error> actual_next = init.transform([]() { return 2; });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // int to void
    expected<int, error> init = make_error();
    expected<void, error> expected_next = make_error();
    expected<void, error> actual_next = init.transform([](int) {});
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to void
    expected<void, error> init = make_error();
    expected<void, error> actual_next = init.transform([]() {});
    EXPECT_TRUE(actual_next == init);
  }
}

// transform_error

TEST(LlvmLibcExpectedTest, TransformError_HasValue) {
  { // int
    expected<int, error> init(1);
    expected<int, fatal_error> actual_next =
        init.transform_error([](error) { return fatal_error(); });
    ASSERT_TRUE(actual_next.has_value());
    EXPECT_TRUE(*actual_next == *init);
  }
  { // void
    expected<void, error> init;
    expected<void, fatal_error> actual_next =
        init.transform_error([](error) { return fatal_error(); });
    ASSERT_TRUE(actual_next.has_value());
  }
}

TEST(LlvmLibcExpectedTest, TransformError_HasError) {
  { // int
    expected<int, error> init = make_error();
    expected<int, fatal_error> actual_next =
        init.transform_error([&](error) { return fatal_error::D; });
    ASSERT_TRUE(!actual_next.has_value());
    ASSERT_EQ(actual_next.error(), fatal_error::D);
  }
  { // void
    expected<void, error> init = make_error();
    expected<void, fatal_error> actual_next =
        init.transform_error([&](error) { return fatal_error::D; });
    ASSERT_TRUE(!actual_next.has_value());
    ASSERT_EQ(actual_next.error(), fatal_error::D);
  }
}

// and_then

TEST(LlvmLibcExpectedTest, AndThen_HasValue) {
  { // int to int
    expected<int, error> init(1);
    expected<int, error> expected_next(2);
    expected<int, error> actual_next =
        init.and_then([&](int) { return expected_next; });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to int
    expected<void, error> init;
    expected<int, error> expected_next(2);
    expected<int, error> actual_next =
        init.and_then([&]() { return expected_next; });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // int to void
    expected<int, error> init(1);
    expected<void, error> expected_next;
    expected<void, error> actual_next =
        init.and_then([&](int) { return expected_next; });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to void
    expected<void, error> init;
    expected<void, error> actual_next = init.and_then([&]() { return init; });
    EXPECT_TRUE(actual_next == init);
  }
}

TEST(LlvmLibcExpectedTest, AndThen_HasError) {
  { // int to int
    expected<int, error> init = make_error();
    expected<int, error> expected_next = make_error();
    expected<int, error> actual_next =
        init.and_then([&](int) { return expected<int, error>(1); });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to int
    expected<void, error> init = make_error();
    expected<int, error> expected_next = make_error();
    expected<int, error> actual_next =
        init.and_then([&]() { return expected<int, error>(1); });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // int to void
    expected<int, error> init = make_error();
    expected<void, error> expected_next = make_error();
    expected<void, error> actual_next =
        init.and_then([&](int) { return expected<void, error>(); });
    EXPECT_TRUE(actual_next == expected_next);
  }
  { // void to void
    expected<void, error> init = make_error();
    expected<void, error> expected_next = make_error();
    expected<void, error> actual_next =
        init.and_then([&]() { return expected<void, error>(); });
    EXPECT_TRUE(actual_next == expected_next);
  }
}

// or_else

TEST(LlvmLibcExpectedTest, OrElse_HasValue) {
  { // int
    using T = expected<int, error>;
    bool called = false;
    T init(1);
    T actual_next = init.or_else([&](error) {
      called = true;
      return T();
    });
    EXPECT_TRUE(actual_next == init);
    EXPECT_FALSE(called);
  }
  { // void
    using T = expected<void, error>;
    bool called = false;
    T init;
    T actual_next = init.or_else([&](error) {
      called = true;
      return T();
    });
    EXPECT_TRUE(actual_next == init);
    EXPECT_FALSE(called);
  }
}

TEST(LlvmLibcExpectedTest, OrElse_HasError) {
  { // int
    bool called = false;
    expected<int, error> init = make_error();
    expected<int, error> expected_next(10);
    expected<int, error> actual_next = init.or_else([&](error) {
      called = true;
      return expected_next;
    });
    EXPECT_TRUE(actual_next == expected_next);
    EXPECT_TRUE(called);
  }
  { // void
    bool called = false;
    expected<void, error> init = make_error();
    expected<void, error> expected_next;
    expected<void, error> actual_next = init.or_else([&](error) {
      called = true;
      return expected_next;
    });
    EXPECT_TRUE(actual_next == expected_next);
    EXPECT_TRUE(called);
  }
}

// transform_error
