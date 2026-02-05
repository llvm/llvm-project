// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatsisf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from int32_t to float
COMPILER_RT_ABI float __floatsisf(int32_t a);

int test__floatsisf(int line, uint32_t a, uint32_t expected_rep) {
  float x = __floatsisf(a);
  int ret = compareResultF(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatsisf(%08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a, toRep32(x), expected_rep);
  }
  return ret;
}

#define test__floatsisf(a, x) test__floatsisf(__LINE__, a, x)

int main(void) {
  int status = 0;

  status |= test__floatsisf(0x00000000, 0x00000000);
  status |= test__floatsisf(0x00000001, 0x3f800000);
  status |= test__floatsisf(0x08000000, 0x4d000000);
  status |= test__floatsisf(0x08000004, 0x4d000000);
  status |= test__floatsisf(0x08000008, 0x4d000000);
  status |= test__floatsisf(0x0800000c, 0x4d000001);
  status |= test__floatsisf(0x08000010, 0x4d000001);
  status |= test__floatsisf(0x08000014, 0x4d000001);
  status |= test__floatsisf(0x08000018, 0x4d000002);
  status |= test__floatsisf(0x0800001c, 0x4d000002);
  status |= test__floatsisf(0x7fffffff, 0x4f000000);
  status |= test__floatsisf(0x80000000, 0xcf000000);
  status |= test__floatsisf(0x80000001, 0xcf000000);
  status |= test__floatsisf(0xf7ffffe4, 0xcd000002);
  status |= test__floatsisf(0xf7ffffe8, 0xcd000002);
  status |= test__floatsisf(0xf7ffffec, 0xcd000001);
  status |= test__floatsisf(0xf7fffff0, 0xcd000001);
  status |= test__floatsisf(0xf7fffff4, 0xcd000001);
  status |= test__floatsisf(0xf7fffff8, 0xcd000000);
  status |= test__floatsisf(0xf7fffffc, 0xcd000000);
  status |= test__floatsisf(0xf8000000, 0xcd000000);

  return status;
}
