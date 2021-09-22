//===-- Unittests for bsearch ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/bsearch.h"

#include "utils/UnitTest/Test.h"

#include <stdlib.h>

static int int_compare(const void *l, const void *r) {
  int li = *reinterpret_cast<const int *>(l);
  int ri = *reinterpret_cast<const int *>(r);
  if (li == ri)
    return 0;
  else if (li > ri)
    return 1;
  else
    return -1;
}

TEST(LlvmLibcBsearchTest, ErrorInputs) {
  int val = 123;
  EXPECT_TRUE(__llvm_libc::bsearch(nullptr, &val, 1, sizeof(int),
                                   int_compare) == nullptr);
  EXPECT_TRUE(__llvm_libc::bsearch(&val, nullptr, 1, sizeof(int),
                                   int_compare) == nullptr);
  EXPECT_TRUE(__llvm_libc::bsearch(&val, &val, 0, sizeof(int), int_compare) ==
              nullptr);
  EXPECT_TRUE(__llvm_libc::bsearch(&val, &val, 1, 0, int_compare) == nullptr);
}

TEST(LlvmLibcBsearchTest, IntegerArray) {
  constexpr int array[25] = {10,   23,   33,    35,   55,   70,   71,
                             100,  110,  123,   133,  135,  155,  170,
                             171,  1100, 1110,  1123, 1133, 1135, 1155,
                             1170, 1171, 11100, 12310};
  constexpr size_t array_size = sizeof(array) / sizeof(int);

  for (size_t s = 1; s <= array_size; ++s) {
    for (size_t i = 0; i < s; ++i) {
      int key = array[i];
      void *elem =
          __llvm_libc::bsearch(&key, array, s, sizeof(int), int_compare);
      ASSERT_EQ(*reinterpret_cast<int *>(elem), key);
    }
  }

  // Non existent keys
  for (size_t s = 1; s <= array_size; ++s) {
    int key = 5;
    ASSERT_TRUE(__llvm_libc::bsearch(&key, &array, s, sizeof(int),
                                     int_compare) == nullptr);

    key = 125;
    ASSERT_TRUE(__llvm_libc::bsearch(&key, &array, s, sizeof(int),
                                     int_compare) == nullptr);

    key = 136;
    ASSERT_TRUE(__llvm_libc::bsearch(&key, &array, s, sizeof(int),
                                     int_compare) == nullptr);
    key = 12345;
    ASSERT_TRUE(__llvm_libc::bsearch(&key, &array, s, sizeof(int),
                                     int_compare) == nullptr);
  }
}

TEST(LlvmLibcBsearchTest, SameKeyAndArray) {
  constexpr int array[5] = {1, 2, 3, 4, 5};
  constexpr size_t array_size = sizeof(array) / sizeof(int);
  void *elem =
      __llvm_libc::bsearch(array, array, array_size, sizeof(int), int_compare);
  EXPECT_EQ(*reinterpret_cast<int *>(elem), array[0]);
}
