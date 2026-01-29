// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunsdisf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from uint64_t to float
COMPILER_RT_ABI float __floatunsdisf(uint64_t a);

int test__floatunsdisf(int line, uint64_t a, uint32_t expected_rep) {
  float x = __floatunsdisf(a);
  int ret = compareResultF(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatunsdisf(%016" PRIx64 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a, toRep32(x), expected_rep);
  }
  return ret;
}

#define test__floatunsdisf(a,x) test__floatunsdisf(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__floatunsdisf(0x0000000000000000, 0x00000000);
  status |= test__floatunsdisf(0x0000000000000001, 0x3f800000);
  status |= test__floatunsdisf(0x0000000008000000, 0x4d000000);
  status |= test__floatunsdisf(0x0000000008000004, 0x4d000000);
  status |= test__floatunsdisf(0x0000000008000008, 0x4d000000);
  status |= test__floatunsdisf(0x000000000800000c, 0x4d000001);
  status |= test__floatunsdisf(0x0000000008000010, 0x4d000001);
  status |= test__floatunsdisf(0x0000000008000014, 0x4d000001);
  status |= test__floatunsdisf(0x0000000008000018, 0x4d000002);
  status |= test__floatunsdisf(0x000000000800001c, 0x4d000002);
  status |= test__floatunsdisf(0x0000082345000000, 0x55023450);
  status |= test__floatunsdisf(0x4000004000000001, 0x5e800001);
  status |= test__floatunsdisf(0x8000000000000000, 0x5f000000);
  status |= test__floatunsdisf(0x8000008000000000, 0x5f000000);
  status |= test__floatunsdisf(0xffffffffffffffff, 0x5f800000);

  return status;
}
