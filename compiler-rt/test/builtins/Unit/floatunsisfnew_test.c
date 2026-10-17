// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunsisf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from uint32_t to float
COMPILER_RT_ABI float __floatunsisf(uint32_t a);

int test__floatunsisf(uint32_t a, uint32_t expected_rep, int line) {
  float x = __floatunsisf(a);
  int ret = compareResultF(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatunsisf(%08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a, toRep32(x), expected_rep);
  }
  return ret;
}

#define test__floatunsisf(a, x) test__floatunsisf(a, x, __LINE__)

int main(void) {
  int status = 0;

  status |= test__floatunsisf(0x00000000, 0x00000000);
  status |= test__floatunsisf(0x00000001, 0x3f800000);
  status |= test__floatunsisf(0x08000000, 0x4d000000);
  status |= test__floatunsisf(0x08000004, 0x4d000000);
  status |= test__floatunsisf(0x08000008, 0x4d000000);
  status |= test__floatunsisf(0x0800000c, 0x4d000001);
  status |= test__floatunsisf(0x08000010, 0x4d000001);
  status |= test__floatunsisf(0x08000014, 0x4d000001);
  status |= test__floatunsisf(0x08000018, 0x4d000002);
  status |= test__floatunsisf(0x0800001c, 0x4d000002);
  status |= test__floatunsisf(0xfffffe00, 0x4f7ffffe);
  status |= test__floatunsisf(0xfffffe7f, 0x4f7ffffe);
  status |= test__floatunsisf(0xfffffe80, 0x4f7ffffe);
  status |= test__floatunsisf(0xfffffe81, 0x4f7fffff);
  status |= test__floatunsisf(0xffffff00, 0x4f7fffff);
  status |= test__floatunsisf(0xffffff7f, 0x4f7fffff);
  status |= test__floatunsisf(0xffffff80, 0x4f800000);
  status |= test__floatunsisf(0xffffff81, 0x4f800000);
  status |= test__floatunsisf(0xffffffff, 0x4f800000);

  return status;
}
