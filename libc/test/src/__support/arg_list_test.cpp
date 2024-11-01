//===-- Unittests for ArgList ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/arg_list.h"

#include "test/UnitTest/Test.h"

int get_nth_int(int n, ...) {
  va_list vlist;
  va_start(vlist, n);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  for (int i = 0; i < n; ++i) {
    v.next_var<int>();
  }
  return v.next_var<int>();
}

TEST(LlvmLibcArgListTest, BasicUsage) {
  ASSERT_EQ(get_nth_int(5, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90), 50);
}

int sum_two_nums(int first, int second, ...) {
  va_list vlist;
  va_start(vlist, second);
  __llvm_libc::internal::ArgList v1(vlist);
  va_end(vlist);

  __llvm_libc::internal::ArgList v2 = v1;

  int first_val;
  for (int i = 0; i < first; ++i) {
    v1.next_var<int>();
  }
  first_val = v1.next_var<int>();

  int second_val;
  for (int i = 0; i < second; ++i) {
    v2.next_var<int>();
  }
  second_val = v2.next_var<int>();

  return first_val + second_val;
}

TEST(LlvmLibcArgListTest, CopyConstructor) {
  ASSERT_EQ(sum_two_nums(3, 1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
            10);

  ASSERT_EQ(sum_two_nums(3, 5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
            40);
}
