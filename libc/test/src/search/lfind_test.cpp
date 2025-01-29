//===-- Unittests for lfind -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/lfind.h"
#include "test/UnitTest/Test.h"

int compar(const void *a, const void *b) {
  return *reinterpret_cast<const int *>(a) != *reinterpret_cast<const int *>(b);
}

TEST(LlvmLibcLfindTest, SearchHead) {
  int list[3] = {1, 2, 3};
  size_t len = 3;
  int key = 1;
  void *ret = LIBC_NAMESPACE::lfind(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == &list[0]);
}

TEST(LlvmLibcLfindTest, SearchMiddle) {
  int list[3] = {1, 2, 3};
  size_t len = 3;
  int key = 2;
  void *ret = LIBC_NAMESPACE::lfind(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == &list[1]);
}

TEST(LlvmLibcLfindTest, SearchTail) {
  int list[3] = {1, 2, 3};
  size_t len = 3;
  int key = 3;
  void *ret = LIBC_NAMESPACE::lfind(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == &list[2]);
}

TEST(LlvmLibcLfindTest, SearchNonExistent) {
  int list[3] = {1, 2, 3};
  size_t len = 3;
  int key = 5;
  void *ret = LIBC_NAMESPACE::lfind(&key, list, &len, sizeof(int), compar);
  ASSERT_TRUE(ret == nullptr);
}
