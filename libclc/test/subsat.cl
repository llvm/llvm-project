//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

__kernel void test_subsat_char(char *a, char x, char y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_uchar(uchar *a, uchar x, uchar y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_long(long *a, long x, long y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_ulong(ulong *a, ulong x, ulong y) {
  *a = sub_sat(x, y);
  return;
}