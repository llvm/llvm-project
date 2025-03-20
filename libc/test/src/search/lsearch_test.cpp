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
  int list[3] = {1, 2, 3};
  size_t len = 3;
  int key = 1;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == &list[0]);
}

TEST(LlvmLibcLsearchTest, SearchMiddle) {
  int list[3] = {1, 2, 3};
  size_t len = 3;
  int key = 2;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == &list[1]);
}

TEST(LlvmLibcLsearchTest, SearchTail) {
  int list[3] = {1, 2, 3};
  size_t len = 3;
  int key = 3;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == &list[2]);
}

TEST(LlvmLibcLsearchTest, SearchNonExistent) {
  int list[4] = {1, 2, 3, 0};
  size_t len = 3;
  int key = 4;
  void *ret = LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == &list[3]);
  ASSERT_EQ(key, list[3]);
  ASSERT_EQ(len, 4UL);
}

TEST(LlvmLibcLsearchTest, SearchExceptional) {
  int list[3] = {1, 2, 3};
  size_t len = 3;
  size_t max_len = ~0;
  int key = 3;

  ASSERT_EQ(LIBC_NAMESPACE::lsearch(nullptr, list, &len, sizeof(int), compar),
            nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::lsearch(&key, nullptr, &len, sizeof(int), compar),
            nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::lsearch(&key, list, nullptr, sizeof(int), compar),
            nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::lsearch(&key, list, &max_len, sizeof(int), compar),
            nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::lsearch(&key, list, &len, sizeof(int), nullptr),
            nullptr);
}
