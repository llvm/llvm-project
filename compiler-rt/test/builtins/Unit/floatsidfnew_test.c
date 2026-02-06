// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatsidf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from int32_t to double
COMPILER_RT_ABI double __floatsidf(int32_t a);

int test__floatsidf(int line, uint32_t a, uint64_t expected_rep) {
  double x = __floatsidf(a);
  int ret = compareResultD(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatsidf(%08" PRIx32 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a, toRep64(x), expected_rep);
  }
  return ret;
}

#define test__floatsidf(a, x) test__floatsidf(__LINE__, a, x)

int main(void) {
  int status = 0;

  status |= test__floatsidf(0x00000000, 0x0000000000000000);
  status |= test__floatsidf(0x00000001, 0x3ff0000000000000);
  status |= test__floatsidf(0x40000200, 0x41d0000080000000);
  status |= test__floatsidf(0x40000400, 0x41d0000100000000);
  status |= test__floatsidf(0x7fffffff, 0x41dfffffffc00000);
  status |= test__floatsidf(0x80000000, 0xc1e0000000000000);
  status |= test__floatsidf(0x80000001, 0xc1dfffffffc00000);

  return status;
}
