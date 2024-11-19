//===-- Unittests for lsearch ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/lsearch.h"
#include "test/UnitTest/Test.h"

int compar(const void *a, const void *b) {
  return *reinterpret_cast<const int *>(a) != *reinterpret_cast<const int *>(b);
}

TEST(LlvmLibcLsearchTest, SearchHead) {
  int list[4] = {1, 2, 3, 4};
  size_t len = 3;
  int key = 1;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);

  ASSERT_EQ(static_cast<int *>(ret), &list[0]);
  ASSERT_EQ(len, static_cast<size_t>(3));
  ASSERT_EQ(list[1], 2);
  ASSERT_EQ(list[2], 3);
  ASSERT_EQ(list[3], 4);
  ASSERT_EQ(list[3], 4);
}

TEST(LlvmLibcLsearchTest, SearchMiddle) {
  int list[4] = {1, 2, 3, 4};
  size_t len = 3;
  int key = 2;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);
  ASSERT_EQ(static_cast<int *>(ret), &list[1]);
  ASSERT_EQ(len, static_cast<size_t>(3));
  ASSERT_EQ(list[0], 1);
  ASSERT_EQ(list[1], 2);
  ASSERT_EQ(list[2], 3);
  ASSERT_EQ(list[3], 4);
}

TEST(LlvmLibcLsearchTest, SearchTail) {
  int list[4] = {1, 2, 3, 4};
  size_t len = 3;
  int key = 3;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);
  ASSERT_EQ(static_cast<int *>(ret), &list[2]);
  ASSERT_EQ(len, static_cast<size_t>(3));
  ASSERT_EQ(list[0], 1);
  ASSERT_EQ(list[1], 2);
  ASSERT_EQ(list[2], 3);
  ASSERT_EQ(list[3], 4);
}

TEST(LlvmLibcLsearchTest, SearchNonExistent) {
  int list[4] = {1, 2, 3, 4};
  size_t len = 3;
  int key = 5;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);

  ASSERT_EQ(static_cast<int *>(ret), &list[3]);
  ASSERT_EQ(len, static_cast<size_t>(4));
  ASSERT_EQ(list[0], 1);
  ASSERT_EQ(list[1], 2);
  ASSERT_EQ(list[2], 3);
  ASSERT_EQ(list[3], 5);
}

TEST(LlvmLibcLsearchTest, SearchNonExistentEmpty) {
  int list[1] = {1};
  size_t len = 0;
  int key = 0;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);

  ASSERT_EQ(static_cast<int *>(ret), &list[0]);
  ASSERT_EQ(len, static_cast<size_t>(1));
  ASSERT_EQ(list[0], 0);
}
