//===-- Unittests for qsort_s ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __STDC_WANT_LIB_EXT1__ 1
#include "QsortReentrantTest.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/errno_t.h"
#include "hdr/types/rsize_t.h"
#include "src/stdlib/qsort_s.h"
#include "test/UnitTest/ConstraintHandlerCheckingTest.h"
#include "test/UnitTest/Test.h"

QSORTREENTRANT_TEST(QsortS, LIBC_NAMESPACE::qsort_s, rsize_t)

using LlvmLibcQsortSConstraintTest =
    LIBC_NAMESPACE::testing::ConstraintHandlerCheckingTest;

static int compare(const void *a, const void *b, void *) {
  const int *a_int = static_cast<const int *>(a);
  const int *b_int = static_cast<const int *>(b);
  return *a_int - *b_int;
}

TEST_F(LlvmLibcQsortSConstraintTest, ArraySizeGreaterThanRsizeMax) {
  int array[] = {1, 3, 2};
  errno_t retval =
      LIBC_NAMESPACE::qsort_s(array, RSIZE_MAX + 1, sizeof(int), compare, 0);
  EXPECT_NE(retval, 0);
  EXPECT_STREQ(buffer, "qsort_s: array_size cannot be greater than RSIZE_MAX");
  EXPECT_EQ(array[0], 1);
  EXPECT_EQ(array[1], 3);
  EXPECT_EQ(array[2], 2);
}

TEST_F(LlvmLibcQsortSConstraintTest, ElemSizeGreaterThanRsizeMax) {
  int array[] = {1, 3, 2};
  errno_t retval = LIBC_NAMESPACE::qsort_s(array, sizeof(array) / sizeof(int),
                                           RSIZE_MAX + 1, compare, 0);
  EXPECT_NE(retval, 0);
  EXPECT_STREQ(buffer, "qsort_s: elem_size cannot be greater than RSIZE_MAX");
  EXPECT_EQ(array[0], 1);
  EXPECT_EQ(array[1], 3);
  EXPECT_EQ(array[2], 2);
}

TEST_F(LlvmLibcQsortSConstraintTest, ArrayPointerIsNull) {
  errno_t retval = LIBC_NAMESPACE::qsort_s(0, 5, sizeof(int), compare, 0);
  EXPECT_NE(retval, 0);
  EXPECT_STREQ(buffer, "qsort_s: if array_size is not equal to zero, then "
                       "neither array nor compare can be a null pointer");
}

TEST_F(LlvmLibcQsortSConstraintTest, ComparePointerIsNull) {
  int array[] = {1, 3, 2};
  errno_t retval = LIBC_NAMESPACE::qsort_s(array, sizeof(array) / sizeof(int),
                                           sizeof(int), 0, 0);
  EXPECT_NE(retval, 0);
  EXPECT_STREQ(buffer, "qsort_s: if array_size is not equal to zero, then "
                       "neither array nor compare can be a null pointer");
  EXPECT_EQ(array[0], 1);
  EXPECT_EQ(array[1], 3);
  EXPECT_EQ(array[2], 2);
}
