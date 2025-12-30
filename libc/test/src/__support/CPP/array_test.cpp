//===-- Unittests for Array -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::array;

TEST(LlvmLibcArrayTest, Basic) {
  array<int, 3> a = {0, 1, 2};

  ASSERT_EQ(a.data(), &a.front());
  ASSERT_EQ(a.front(), 0);
  ASSERT_EQ(a.back(), 2);
  ASSERT_EQ(a.size(), size_t{3});
  ASSERT_EQ(a[1], 1);
  ASSERT_FALSE(a.empty());
  ASSERT_NE(a.begin(), a.end());
  ASSERT_EQ(*a.begin(), a.front());

  auto it = a.rbegin();
  ASSERT_EQ(*it, 2);
  ASSERT_EQ(*(++it), 1);
  ASSERT_EQ(*(++it), 0);

  for (int &x : a)
    ASSERT_GE(x, 0);
}

// Test const_iterator and const variant methods.
TEST(LlvmLibcArrayTest, Const) {
  const array<int, 3> z = {3, 4, 5};

  ASSERT_EQ(3, z.front());
  ASSERT_EQ(4, z[1]);
  ASSERT_EQ(5, z.back());
  ASSERT_EQ(3, *z.data());

  // begin, cbegin, end, cend
  array<int, 3>::const_iterator it2 = z.begin();
  ASSERT_EQ(*it2, z.front());
  it2 = z.cbegin();
  ASSERT_EQ(*it2, z.front());
  it2 = z.end();
  ASSERT_NE(it2, z.begin());
  it2 = z.cend();
  ASSERT_NE(it2, z.begin());

  // rbegin, crbegin, rend, crend
  array<int, 3>::const_reverse_iterator it = z.rbegin();
  ASSERT_EQ(*it, z.back());
  it = z.crbegin();
  ASSERT_EQ(*it, z.back());
  it = z.rend();
  ASSERT_EQ(*--it, z.front());
  it = z.crend();
  ASSERT_EQ(*--it, z.front());

  for (const int &x : z)
    ASSERT_GE(x, 0);
}
