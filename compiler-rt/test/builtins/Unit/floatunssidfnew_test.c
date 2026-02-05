// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunssidf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from uint32_t to double
COMPILER_RT_ABI double __floatunssidf(uint32_t a);

int test__floatunssidf(int line, uint32_t a, uint64_t expected_rep) {
  double x = __floatunssidf(a);
  int ret = compareResultD(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatunssidf(%08" PRIx32 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a, toRep64(x), expected_rep);
  }
  return ret;
}

#define test__floatunssidf(a, x) test__floatunssidf(__LINE__, a, x)

int main(void) {
  int status = 0;

  status |= test__floatunssidf(0x00000000, 0x0000000000000000);
  status |= test__floatunssidf(0x00000001, 0x3ff0000000000000);
  status |= test__floatunssidf(0x80000400, 0x41e0000080000000);
  status |= test__floatunssidf(0x80000800, 0x41e0000100000000);
  status |= test__floatunssidf(0xffffffff, 0x41efffffffe00000);

  return status;
}
