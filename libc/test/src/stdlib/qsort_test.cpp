//===-- Unittests for qsort -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort.h"

#include "test/UnitTest/Test.h"

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

TEST(LlvmLibcQsortTest, SortedArray) {
  int array[25] = {10,   23,   33,   35,   55,   70,    71,   100,  110,
                   123,  133,  135,  155,  170,  171,   1100, 1110, 1123,
                   1133, 1135, 1155, 1170, 1171, 11100, 12310};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 10);
  ASSERT_LE(array[1], 23);
  ASSERT_LE(array[2], 33);
  ASSERT_LE(array[3], 35);
  ASSERT_LE(array[4], 55);
  ASSERT_LE(array[5], 70);
  ASSERT_LE(array[6], 71);
  ASSERT_LE(array[7], 100);
  ASSERT_LE(array[8], 110);
  ASSERT_LE(array[9], 123);
  ASSERT_LE(array[10], 133);
  ASSERT_LE(array[11], 135);
  ASSERT_LE(array[12], 155);
  ASSERT_LE(array[13], 170);
  ASSERT_LE(array[14], 171);
  ASSERT_LE(array[15], 1100);
  ASSERT_LE(array[16], 1110);
  ASSERT_LE(array[17], 1123);
  ASSERT_LE(array[18], 1133);
  ASSERT_LE(array[19], 1135);
  ASSERT_LE(array[20], 1155);
  ASSERT_LE(array[21], 1170);
  ASSERT_LE(array[22], 1171);
  ASSERT_LE(array[23], 11100);
  ASSERT_LE(array[24], 12310);
}

TEST(LlvmLibcQsortTest, ReverseSortedArray) {
  int array[25] = {25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                   12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  for (int i = 0; i < int(ARRAY_SIZE - 1); ++i)
    ASSERT_LE(array[i], i + 1);
}

TEST(LlvmLibcQsortTest, AllEqualElements) {
  int array[25] = {100, 100, 100, 100, 100, 100, 100, 100, 100,
                   100, 100, 100, 100, 100, 100, 100, 100, 100,
                   100, 100, 100, 100, 100, 100, 100};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  for (size_t i = 0; i < ARRAY_SIZE - 1; ++i)
    ASSERT_LE(array[i], 100);
}

TEST(LlvmLibcQsortTest, UnsortedArray1) {
  int array[25] = {10, 23,  8,  35, 55, 45, 40,  100,  110,  123,  90, 80,  70,
                   60, 171, 11, 1,  -1, -5, -10, 1155, 1170, 1171, 12, -100};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], -100);
  ASSERT_LE(array[1], -10);
  ASSERT_LE(array[2], -5);
  ASSERT_LE(array[3], -1);
  ASSERT_LE(array[4], 1);
  ASSERT_LE(array[5], 8);
  ASSERT_LE(array[6], 10);
  ASSERT_LE(array[7], 11);
  ASSERT_LE(array[8], 12);
  ASSERT_LE(array[9], 23);
  ASSERT_LE(array[10], 35);
  ASSERT_LE(array[11], 40);
  ASSERT_LE(array[12], 45);
  ASSERT_LE(array[13], 55);
  ASSERT_LE(array[14], 60);
  ASSERT_LE(array[15], 70);
  ASSERT_LE(array[16], 80);
  ASSERT_LE(array[17], 90);
  ASSERT_LE(array[18], 100);
  ASSERT_LE(array[19], 110);
  ASSERT_LE(array[20], 123);
  ASSERT_LE(array[21], 171);
  ASSERT_LE(array[22], 1155);
  ASSERT_LE(array[23], 1170);
  ASSERT_LE(array[24], 1171);
}

TEST(LlvmLibcQsortTest, UnsortedArray2) {
  int array[7] = {10, 40, 45, 55, 35, 23, 60};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 10);
  ASSERT_LE(array[1], 23);
  ASSERT_LE(array[2], 35);
  ASSERT_LE(array[3], 40);
  ASSERT_LE(array[4], 45);
  ASSERT_LE(array[5], 55);
  ASSERT_LE(array[6], 60);
}

TEST(LlvmLibcQsortTest, UnsortedArrayDuplicateElements1) {
  int array[6] = {10, 10, 20, 20, 5, 5};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 5);
  ASSERT_LE(array[1], 5);
  ASSERT_LE(array[2], 10);
  ASSERT_LE(array[3], 10);
  ASSERT_LE(array[4], 20);
  ASSERT_LE(array[5], 20);
}

TEST(LlvmLibcQsortTest, UnsortedArrayDuplicateElements2) {
  int array[10] = {20, 10, 10, 10, 10, 20, 21, 21, 21, 21};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 10);
  ASSERT_LE(array[1], 10);
  ASSERT_LE(array[2], 10);
  ASSERT_LE(array[3], 10);
  ASSERT_LE(array[4], 20);
  ASSERT_LE(array[5], 20);
  ASSERT_LE(array[6], 21);
  ASSERT_LE(array[7], 21);
  ASSERT_LE(array[8], 21);
  ASSERT_LE(array[9], 21);
}

TEST(LlvmLibcQsortTest, UnsortedArrayDuplicateElements3) {
  int array[10] = {20, 30, 30, 30, 30, 20, 21, 21, 21, 21};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 20);
  ASSERT_LE(array[1], 20);
  ASSERT_LE(array[2], 21);
  ASSERT_LE(array[3], 21);
  ASSERT_LE(array[4], 21);
  ASSERT_LE(array[5], 21);
  ASSERT_LE(array[6], 30);
  ASSERT_LE(array[7], 30);
  ASSERT_LE(array[8], 30);
  ASSERT_LE(array[9], 30);
}

TEST(LlvmLibcQsortTest, UnsortedThreeElementArray1) {
  int array[3] = {14999024, 0, 3};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 0);
  ASSERT_LE(array[1], 3);
  ASSERT_LE(array[2], 14999024);
}

TEST(LlvmLibcQsortTest, UnsortedThreeElementArray2) {
  int array[3] = {3, 14999024, 0};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 0);
  ASSERT_LE(array[1], 3);
  ASSERT_LE(array[2], 14999024);
}

TEST(LlvmLibcQsortTest, UnsortedThreeElementArray3) {
  int array[3] = {3, 0, 14999024};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 0);
  ASSERT_LE(array[1], 3);
  ASSERT_LE(array[2], 14999024);
}

TEST(LlvmLibcQsortTest, SameElementThreeElementArray) {
  int array[3] = {12345, 12345, 12345};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 12345);
  ASSERT_LE(array[1], 12345);
  ASSERT_LE(array[2], 12345);
}

TEST(LlvmLibcQsortTest, UnsortedTwoElementArray1) {
  int array[2] = {14999024, 0};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 0);
  ASSERT_LE(array[1], 14999024);
}

TEST(LlvmLibcQsortTest, UnsortedTwoElementArray2) {
  int array[2] = {0, 14999024};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 0);
  ASSERT_LE(array[1], 14999024);
}

TEST(LlvmLibcQsortTest, SameElementTwoElementArray) {
  int array[2] = {12345, 12345};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], 12345);
  ASSERT_LE(array[1], 12345);
}

TEST(LlvmLibcQSortTest, SingleElementArray) {
  constexpr int ELEM = 12345;
  int array[1] = {ELEM};
  constexpr size_t ARRAY_SIZE = sizeof(array) / sizeof(int);

  __llvm_libc::qsort(array, ARRAY_SIZE, sizeof(int), int_compare);

  ASSERT_LE(array[0], ELEM);
}
