//===-- Unittests for span ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/span.h"
#include "test/UnitTest/Test.h"

using __llvm_libc::cpp::array;
using __llvm_libc::cpp::span;

TEST(LlvmLibcSpanTest, InitializeEmpty) {
  span<int> s;
  ASSERT_EQ(s.size(), size_t(0));
  ASSERT_TRUE(s.empty());
  ASSERT_TRUE(s.data() == nullptr);
}

TEST(LlvmLibcSpanTest, InitializeSingleton) {
  int a = 42;
  span<int> s(&a, 1);
  ASSERT_EQ(s.size(), size_t(1));
  ASSERT_FALSE(s.empty());
  ASSERT_TRUE(s.data() == &a);
  ASSERT_EQ(s.front(), 42);
  ASSERT_EQ(s.back(), 42);
}

TEST(LlvmLibcSpanTest, InitializeCArray) {
  int a[] = {1, 2, 3};
  span<int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_FALSE(s.empty());
  ASSERT_TRUE(s.data() == &a[0]);
  ASSERT_EQ(s.front(), 1);
  ASSERT_EQ(s.back(), 3);
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeArray) {
  array<int, 3> a = {1, 2, 3};
  span<int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_FALSE(s.empty());
  ASSERT_TRUE(s.data() == &a[0]);
  ASSERT_EQ(s.front(), 1);
  ASSERT_EQ(s.back(), 3);
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeViewFormMutableSingleton) {
  int a = 42;
  span<const int> s(&a, 1);
  ASSERT_EQ(s.size(), size_t(1));
  ASSERT_TRUE(s.data() == &a);
}

TEST(LlvmLibcSpanTest, InitializeViewFormMutableCArray) {
  int a[] = {1, 2, 3};
  span<const int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeViewFormMutableArray) {
  array<int, 3> a = {1, 2, 3};
  span<const int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeFromMutable) {
  span<int> s;
  span<const int> view(s);
  (void)view;
}

TEST(LlvmLibcSpanTest, Assign) {
  span<int> s;
  span<int> other;
  other = s;
}

TEST(LlvmLibcSpanTest, AssignFromMutable) {
  span<int> s;
  span<const int> view;
  view = s;
}

TEST(LlvmLibcSpanTest, Modify) {
  int a[] = {1, 2, 3};
  span<int> s(a);
  for (int &value : s)
    ++value;
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s[0], 2);
  ASSERT_EQ(s[1], 3);
  ASSERT_EQ(s[2], 4);
}

TEST(LlvmLibcSpanTest, SubSpan) {
  int a[] = {1, 2, 3};
  span<const int> s(a);
  { // same span
    const auto _ = s.subspan(0);
    ASSERT_EQ(_.size(), size_t(3));
    ASSERT_EQ(_[0], 1);
    ASSERT_EQ(_[1], 2);
    ASSERT_EQ(_[2], 3);
  }
  { // last element
    const auto _ = s.subspan(2);
    ASSERT_EQ(_.size(), size_t(1));
    ASSERT_EQ(_[0], 3);
  }
  { // no element
    const auto _ = s.subspan(3);
    ASSERT_EQ(_.size(), size_t(0));
  }
  { // first element
    const auto _ = s.subspan(0, 1);
    ASSERT_EQ(_.size(), size_t(1));
    ASSERT_EQ(_[0], 1);
  }
}

TEST(LlvmLibcSpanTest, FirstAndLastSubSpan) {
  int a[] = {1, 2, 3};
  span<const int> s(a);

  const auto first = s.first(2);
  ASSERT_EQ(first.size(), size_t(2));
  ASSERT_EQ(first[0], 1);
  ASSERT_EQ(first[1], 2);

  const auto last = s.last(2);
  ASSERT_EQ(last.size(), size_t(2));
  ASSERT_EQ(last[0], 2);
  ASSERT_EQ(last[1], 3);
}
