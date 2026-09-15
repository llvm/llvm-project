//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for vector.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/vector.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::vector;

TEST(LlvmLibcVectorTest, InitializeEmpty) {
  vector<int> v;
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), size_t(0));
  EXPECT_EQ(v.capacity(), size_t(0));
  EXPECT_EQ(v.data(), static_cast<int *>(nullptr));
}

TEST(LlvmLibcVectorTest, PushBackAndAccess) {
  vector<int> v;
  ASSERT_TRUE(v.push_back(10));
  ASSERT_TRUE(v.push_back(20));
  ASSERT_TRUE(v.push_back(30));

  EXPECT_FALSE(v.empty());
  EXPECT_EQ(v.size(), size_t(3));
  EXPECT_GE(v.capacity(), size_t(3));

  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
  EXPECT_EQ(v.front(), 10);
  EXPECT_EQ(v.back(), 30);
  ASSERT_NE(v.data(), nullptr);
  EXPECT_EQ(v.data()[1], 20);
}

TEST(LlvmLibcVectorTest, RangeBasedForLoop) {
  vector<int> v;
  for (int i = 0; i < 5; ++i)
    ASSERT_TRUE(v.push_back(i * 10));

  int expected = 0;
  for (int val : v) {
    EXPECT_EQ(val, expected);
    expected += 10;
  }
  EXPECT_EQ(expected, 50);

  // Test cbegin / cend
  int const_expected = 0;
  for (auto it = v.cbegin(); it != v.cend(); ++it) {
    EXPECT_EQ(*it, const_expected);
    const_expected += 10;
  }
  EXPECT_EQ(const_expected, 50);
}

TEST(LlvmLibcVectorTest, MemberTypesAndMaxSize) {
  static_assert(LIBC_NAMESPACE::cpp::is_same_v<vector<int>::value_type, int>);
  static_assert(LIBC_NAMESPACE::cpp::is_same_v<vector<int>::size_type, size_t>);
  static_assert(
      LIBC_NAMESPACE::cpp::is_same_v<vector<int>::difference_type, ptrdiff_t>);
  static_assert(LIBC_NAMESPACE::cpp::is_same_v<vector<int>::pointer, int *>);
  static_assert(
      LIBC_NAMESPACE::cpp::is_same_v<vector<int>::const_pointer, const int *>);

  vector<int> v;
  EXPECT_GT(v.max_size(), size_t(0));
  EXPECT_EQ(v.max_size(), static_cast<size_t>(-1) / sizeof(int));
}

TEST(LlvmLibcVectorTest, PopBack) {
  vector<int> v;
  ASSERT_TRUE(v.push_back(1));
  ASSERT_TRUE(v.push_back(2));
  EXPECT_EQ(v.size(), size_t(2));

  v.pop_back();
  EXPECT_EQ(v.size(), size_t(1));
  EXPECT_EQ(v.back(), 1);

  v.pop_back();
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), size_t(0));
}

TEST(LlvmLibcVectorTest, ReserveAndShrinkToFit) {
  vector<int> v;
  ASSERT_TRUE(v.reserve(100));
  EXPECT_GE(v.capacity(), size_t(100));
  EXPECT_EQ(v.size(), size_t(0));

  ASSERT_TRUE(v.push_back(42));
  EXPECT_EQ(v.size(), size_t(1));
  EXPECT_GE(v.capacity(), size_t(100));

  v.shrink_to_fit();
  EXPECT_EQ(v.size(), size_t(1));
  EXPECT_EQ(v.capacity(), size_t(1));
  EXPECT_EQ(v[0], 42);
}

TEST(LlvmLibcVectorTest, ClearAndReset) {
  vector<int> v;
  for (int i = 0; i < 10; ++i)
    ASSERT_TRUE(v.push_back(i));

  size_t old_cap = v.capacity();
  v.clear();
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), size_t(0));
  EXPECT_EQ(v.capacity(), old_cap);

  // Can reuse buffer after clear
  ASSERT_TRUE(v.push_back(99));
  EXPECT_EQ(v.size(), size_t(1));
  EXPECT_EQ(v[0], 99);

  // Reset frees buffer
  v.reset();
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), size_t(0));
  EXPECT_EQ(v.capacity(), size_t(0));
  EXPECT_EQ(v.data(), static_cast<int *>(nullptr));
}

TEST(LlvmLibcVectorTest, Resize) {
  vector<int> v;
  ASSERT_TRUE(v.resize(5, 7));
  EXPECT_EQ(v.size(), size_t(5));
  for (size_t i = 0; i < 5; ++i)
    EXPECT_EQ(v[i], 7);

  ASSERT_TRUE(v.resize(2));
  EXPECT_EQ(v.size(), size_t(2));
  EXPECT_EQ(v[0], 7);
  EXPECT_EQ(v[1], 7);

  ASSERT_TRUE(v.resize(4, 9));
  EXPECT_EQ(v.size(), size_t(4));
  EXPECT_EQ(v[0], 7);
  EXPECT_EQ(v[1], 7);
  EXPECT_EQ(v[2], 9);
  EXPECT_EQ(v[3], 9);
}

TEST(LlvmLibcVectorTest, MoveOperations) {
  vector<int> v1;
  ASSERT_TRUE(v1.push_back(100));
  ASSERT_TRUE(v1.push_back(200));

  // Move constructor
  vector<int> v2(LIBC_NAMESPACE::cpp::move(v1));
  EXPECT_EQ(v2.size(), size_t(2));
  EXPECT_EQ(v2[0], 100);
  EXPECT_EQ(v2[1], 200);
  EXPECT_TRUE(v1.empty());
  EXPECT_EQ(v1.data(), static_cast<int *>(nullptr));

  // Move assignment
  vector<int> v3;
  v3 = LIBC_NAMESPACE::cpp::move(v2);
  EXPECT_EQ(v3.size(), size_t(2));
  EXPECT_EQ(v3[0], 100);
  EXPECT_EQ(v3[1], 200);
  EXPECT_TRUE(v2.empty());
  EXPECT_EQ(v2.data(), static_cast<int *>(nullptr));
}

TEST(LlvmLibcVectorTest, SwapOperations) {
  vector<int> a;
  ASSERT_TRUE(a.push_back(1));
  ASSERT_TRUE(a.push_back(2));

  vector<int> b;
  ASSERT_TRUE(b.push_back(30));
  ASSERT_TRUE(b.push_back(40));
  ASSERT_TRUE(b.push_back(50));

  // Member swap
  a.swap(b);
  EXPECT_EQ(a.size(), size_t(3));
  EXPECT_EQ(a[0], 30);
  EXPECT_EQ(a[1], 40);
  EXPECT_EQ(a[2], 50);
  EXPECT_EQ(b.size(), size_t(2));
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[1], 2);

  // Free swap
  LIBC_NAMESPACE::cpp::swap(a, b);
  EXPECT_EQ(a.size(), size_t(2));
  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[1], 2);
  EXPECT_EQ(b.size(), size_t(3));
  EXPECT_EQ(b[0], 30);
  EXPECT_EQ(b[1], 40);
  EXPECT_EQ(b[2], 50);
}

struct LifetimeTracker {
  static int constructed;
  static int destructed;
  static int destroy_log[64];
  static int destroy_log_len;
  int value = 0;
  bool moved_from = false;

  LifetimeTracker() : value(0) { ++constructed; }
  explicit LifetimeTracker(int val) : value(val) { ++constructed; }
  LifetimeTracker(const LifetimeTracker &other) : value(other.value) {
    ++constructed;
  }
  LifetimeTracker(LifetimeTracker &&other) noexcept : value(other.value) {
    other.moved_from = true;
    ++constructed;
  }
  ~LifetimeTracker() {
    ++destructed;
    if (!moved_from && destroy_log_len < 64)
      destroy_log[destroy_log_len++] = value;
  }
};

int LifetimeTracker::constructed = 0;
int LifetimeTracker::destructed = 0;
int LifetimeTracker::destroy_log[64] = {};
int LifetimeTracker::destroy_log_len = 0;

TEST(LlvmLibcVectorTest, NonTrivialGrowthAndDestructionOrder) {
  LifetimeTracker::constructed = 0;
  LifetimeTracker::destructed = 0;
  LifetimeTracker::destroy_log_len = 0;

  {
    vector<LifetimeTracker> v;
    // Insert 20 elements to force reallocation past DEFAULT_INITIAL_CAPACITY
    // (16)
    for (int i = 0; i < 20; ++i)
      ASSERT_TRUE(v.emplace_back(i));

    EXPECT_EQ(v.size(), size_t(20));
    EXPECT_GE(v.capacity(), size_t(20));
    for (size_t i = 0; i < 20; ++i)
      EXPECT_EQ(v[i].value, int(i));

    v.shrink_to_fit();
    EXPECT_EQ(v.capacity(), size_t(20));
    for (size_t i = 0; i < 20; ++i)
      EXPECT_EQ(v[i].value, int(i));

    v.pop_back();
    EXPECT_EQ(v.size(), size_t(19));
  }

  EXPECT_EQ(LifetimeTracker::constructed, LifetimeTracker::destructed);
  // First destroyed was element 19 (via pop_back), then 18 down to 0 (LIFO)
  ASSERT_EQ(LifetimeTracker::destroy_log_len, 20);
  EXPECT_EQ(LifetimeTracker::destroy_log[0], 19);
  for (int i = 1; i < 20; ++i)
    EXPECT_EQ(LifetimeTracker::destroy_log[i], 19 - i);
}

TEST(LlvmLibcVectorTest, SelfReferentialPushBackAndResize) {
  vector<int> v;
  // Fill up to initial capacity (16) so the next push_back reallocates.
  for (int i = 0; i < 16; ++i)
    ASSERT_TRUE(v.push_back(i + 100));

  EXPECT_EQ(v.size(), v.capacity());
  // Self-referential push_back across reallocation boundary
  ASSERT_TRUE(v.push_back(v[0]));
  EXPECT_EQ(v.size(), size_t(17));
  EXPECT_EQ(v.back(), 100);

  // Self-referential resize across reallocation boundary
  ASSERT_TRUE(v.resize(40, v[1]));
  EXPECT_EQ(v.size(), size_t(40));
  EXPECT_EQ(v[39], 101);
}

TEST(LlvmLibcVectorTest, StringPointerVector) {
  vector<const char *> ptrs;
  ASSERT_TRUE(ptrs.push_back("first"));
  ASSERT_TRUE(ptrs.push_back("second"));
  ASSERT_TRUE(ptrs.push_back(nullptr));

  EXPECT_EQ(ptrs.size(), size_t(3));
  EXPECT_STREQ(ptrs[0], "first");
  EXPECT_STREQ(ptrs[1], "second");
  EXPECT_EQ(ptrs[2], nullptr);
  EXPECT_EQ(ptrs.data()[2], nullptr);
}
