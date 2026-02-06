// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunssisf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from uint32_t to float
COMPILER_RT_ABI float __floatunssisf(uint32_t a);

int test__floatunssisf(int line, uint32_t a, uint32_t expected_rep) {
  float x = __floatunssisf(a);
  int ret = compareResultF(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatunssisf(%08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a, toRep32(x), expected_rep);
  }
  return ret;
}

#define test__floatunssisf(a, x) test__floatunssisf(__LINE__, a, x)

int main(void) {
  int status = 0;

  status |= test__floatunssisf(0x00000000, 0x00000000);
  status |= test__floatunssisf(0x00000001, 0x3f800000);
  status |= test__floatunssisf(0x08000000, 0x4d000000);
  status |= test__floatunssisf(0x08000004, 0x4d000000);
  status |= test__floatunssisf(0x08000008, 0x4d000000);
  status |= test__floatunssisf(0x0800000c, 0x4d000001);
  status |= test__floatunssisf(0x08000010, 0x4d000001);
  status |= test__floatunssisf(0x08000014, 0x4d000001);
  status |= test__floatunssisf(0x08000018, 0x4d000002);
  status |= test__floatunssisf(0x0800001c, 0x4d000002);
  status |= test__floatunssisf(0xfffffe00, 0x4f7ffffe);
  status |= test__floatunssisf(0xfffffe7f, 0x4f7ffffe);
  status |= test__floatunssisf(0xfffffe80, 0x4f7ffffe);
  status |= test__floatunssisf(0xfffffe81, 0x4f7fffff);
  status |= test__floatunssisf(0xffffff00, 0x4f7fffff);
  status |= test__floatunssisf(0xffffff7f, 0x4f7fffff);
  status |= test__floatunssisf(0xffffff80, 0x4f800000);
  status |= test__floatunssisf(0xffffff81, 0x4f800000);
  status |= test__floatunssisf(0xffffffff, 0x4f800000);

  return status;
}
