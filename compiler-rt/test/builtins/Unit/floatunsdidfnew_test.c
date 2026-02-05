// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunsdidf

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a converted from uint64_t to double
COMPILER_RT_ABI double __floatunsdidf(uint64_t a);

int test__floatunsdidf(int line, uint64_t a, uint64_t expected_rep) {
  double x = __floatunsdidf(a);
  int ret = compareResultD(x, expected_rep);

  if (ret) {
    printf("error at line %d: __floatunsdidf(%016" PRIx64 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a, toRep64(x), expected_rep);
  }
  return ret;
}

#define test__floatunsdidf(a, x) test__floatunsdidf(__LINE__, a, x)

int main(void) {
  int status = 0;

  status |= test__floatunsdidf(0x0000000000000000, 0x0000000000000000);
  status |= test__floatunsdidf(0x0000000000000001, 0x3ff0000000000000);
  status |= test__floatunsdidf(0x0000000080000000, 0x41e0000000000000);
  status |= test__floatunsdidf(0x0000000080000001, 0x41e0000000200000);
  status |= test__floatunsdidf(0x0000000080000003, 0x41e0000000600000);
  status |= test__floatunsdidf(0x0000000080000007, 0x41e0000000e00000);
  status |= test__floatunsdidf(0x00000000fffffff8, 0x41efffffff000000);
  status |= test__floatunsdidf(0x00000000fffffffc, 0x41efffffff800000);
  status |= test__floatunsdidf(0x00000000fffffffe, 0x41efffffffc00000);
  status |= test__floatunsdidf(0x00000000ffffffff, 0x41efffffffe00000);
  status |= test__floatunsdidf(0x0000082345670000, 0x42a0468ace000000);
  status |= test__floatunsdidf(0x0013d0942cab7317, 0x4333d0942cab7317);
  status |= test__floatunsdidf(0x0100000000000000, 0x4370000000000000);
  status |= test__floatunsdidf(0x0100000000000004, 0x4370000000000000);
  status |= test__floatunsdidf(0x0100000000000008, 0x4370000000000000);
  status |= test__floatunsdidf(0x010000000000000c, 0x4370000000000001);
  status |= test__floatunsdidf(0x0100000000000010, 0x4370000000000001);
  status |= test__floatunsdidf(0x0100000000000014, 0x4370000000000001);
  status |= test__floatunsdidf(0x0100000000000018, 0x4370000000000002);
  status |= test__floatunsdidf(0x010000000000001c, 0x4370000000000002);
  status |= test__floatunsdidf(0xffffffffffffffff, 0x43f0000000000000);

  return status;
}
