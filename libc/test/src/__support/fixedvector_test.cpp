//===-- Unittests for FixedVector -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/fixedvector.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcFixedVectorTest, PushAndPop) {
  LIBC_NAMESPACE::FixedVector<int, 20> fixed_vector;
  ASSERT_TRUE(fixed_vector.empty());
  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(fixed_vector.push_back(i));
  ASSERT_FALSE(fixed_vector.empty());
  ASSERT_FALSE(fixed_vector.push_back(123));
  for (int i = 20; i > 0; --i) {
    ASSERT_EQ(fixed_vector.back(), i - 1);
    ASSERT_TRUE(fixed_vector.pop_back());
  }
  ASSERT_FALSE(fixed_vector.pop_back());
  ASSERT_TRUE(fixed_vector.empty());
}

TEST(LlvmLibcFixedVectorTest, Reset) {
  LIBC_NAMESPACE::FixedVector<int, 20> fixed_vector;
  ASSERT_TRUE(fixed_vector.empty());
  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(fixed_vector.push_back(i));
  ASSERT_FALSE(fixed_vector.empty());
  fixed_vector.reset();
  ASSERT_TRUE(fixed_vector.empty());
}

TEST(LlvmLibcFixedVectorTest, Destroy) {
  LIBC_NAMESPACE::FixedVector<int, 20> fixed_vector;
  ASSERT_TRUE(fixed_vector.empty());
  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(fixed_vector.push_back(i));
  ASSERT_FALSE(fixed_vector.empty());
  LIBC_NAMESPACE::FixedVector<int, 20>::destroy(&fixed_vector);
  ASSERT_TRUE(fixed_vector.empty());
}

TEST(LlvmLibcFixedVectorTest, Iteration) {
  LIBC_NAMESPACE::FixedVector<int, 20> v;
  for (int i = 0; i < 3; i++)
    v.push_back(i);
  auto it = v.rbegin();
  ASSERT_EQ(*it, 2);
  ASSERT_EQ(*++it, 1);
  ASSERT_EQ(*++it, 0);
  // TODO: need an overload of Test::test for iterators?
  // ASSERT_EQ(++it, v.rend());
  // ASSERT_EQ(v.rbegin(), v.rbegin());
  ASSERT_TRUE(++it == v.rend());
  for (auto it = v.rbegin(), e = v.rend(); it != e; ++it)
    ASSERT_GT(*it, -1);

  auto forward_it = v.begin();
  ASSERT_EQ(*forward_it, 0);
  ASSERT_EQ(*++forward_it, 1);
  ASSERT_EQ(*++forward_it, 2);
  ASSERT_TRUE(++forward_it == v.end());
  for (auto it = v.begin(), e = v.end(); it != e; ++it)
    ASSERT_GT(*it, -1);
  for (int &x : v)
    ASSERT_GE(x, 0);
}

TEST(LlvmLibcFixedVectorTest, ConstructionFromIterators) {
  LIBC_NAMESPACE::cpp::array<int, 4> arr{1, 2, 3, 4};
  LIBC_NAMESPACE::FixedVector<int, 5> vec(arr.begin(), arr.end());
  ASSERT_EQ(vec.size(), arr.size());
  for (size_t i = 0; i < arr.size(); ++i)
    ASSERT_EQ(vec[i], arr[i]);
}

TEST(LlvmLibcFixedVectorTest, ConstructionFromCountAndValue) {
  constexpr int kVal = 10;
  LIBC_NAMESPACE::FixedVector<int, 5> vec(4, kVal);
  ASSERT_EQ(vec.size(), size_t(4));
  for (size_t i = 0; i < vec.size(); ++i)
    ASSERT_EQ(vec[i], kVal);
}

TEST(LlvmLibcFixedVectorTest, ForwardIteration) {
  LIBC_NAMESPACE::cpp::array<int, 4> arr{1, 2, 3, 4};
  LIBC_NAMESPACE::FixedVector<int, 5> vec(arr.begin(), arr.end());
  ASSERT_EQ(vec.size(), arr.size());
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    auto idx = it - vec.begin();
    ASSERT_EQ(*it, arr[idx]);
  }
}

TEST(LlvmLibcFixedVectorTest, ConstForwardIteration) {
  const LIBC_NAMESPACE::cpp::array<int, 4> arr{1, 2, 3, 4};
  const LIBC_NAMESPACE::FixedVector<int, 5> vec(arr.begin(), arr.end());
  ASSERT_EQ(vec.size(), arr.size());
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    auto idx = it - vec.begin();
    ASSERT_EQ(*it, arr[idx]);
  }
}
