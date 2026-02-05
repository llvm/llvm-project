// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatdisf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from int64_t to float
COMPILER_RT_ABI float __floatdisf(int64_t a);

int test__floatdisf(int line, uint64_t a, uint32_t expected_rep) {
  float x = __floatdisf(a);
  int ret = compareResultF(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatdisf(%016" PRIx64 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a, toRep32(x), expected_rep);
  }
  return ret;
}

#define test__floatdisf(a, x) test__floatdisf(__LINE__, a, x)

int main(void) {
  int status = 0;

  status |= test__floatdisf(0x0000000000000000, 0x00000000);
  status |= test__floatdisf(0x0000000000000001, 0x3f800000);
  status |= test__floatdisf(0x0000000008000000, 0x4d000000);
  status |= test__floatdisf(0x0000000008000004, 0x4d000000);
  status |= test__floatdisf(0x0000000008000008, 0x4d000000);
  status |= test__floatdisf(0x000000000800000c, 0x4d000001);
  status |= test__floatdisf(0x0000000008000010, 0x4d000001);
  status |= test__floatdisf(0x0000000008000014, 0x4d000001);
  status |= test__floatdisf(0x0000000008000018, 0x4d000002);
  status |= test__floatdisf(0x000000000800001c, 0x4d000002);
  status |= test__floatdisf(0x0000082345000000, 0x55023450);
  status |= test__floatdisf(0x4000004000000001, 0x5e800001);
  status |= test__floatdisf(0x7fffffffffffffff, 0x5f000000);
  status |= test__floatdisf(0x8000000000000000, 0xdf000000);
  status |= test__floatdisf(0x8000000000000001, 0xdf000000);
  status |= test__floatdisf(0xfffffffff7ffffe4, 0xcd000002);
  status |= test__floatdisf(0xfffffffff7ffffe8, 0xcd000002);
  status |= test__floatdisf(0xfffffffff7ffffec, 0xcd000001);
  status |= test__floatdisf(0xfffffffff7fffff0, 0xcd000001);
  status |= test__floatdisf(0xfffffffff7fffff4, 0xcd000001);
  status |= test__floatdisf(0xfffffffff7fffff8, 0xcd000000);
  status |= test__floatdisf(0xfffffffff7fffffc, 0xcd000000);
  status |= test__floatdisf(0xfffffffff8000000, 0xcd000000);

  return status;
}
