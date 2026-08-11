// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_addsf3

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// Returns: a + b
COMPILER_RT_ABI float __addsf3(float a, float b);

int test__addsf3(uint32_t a_rep, uint32_t b_rep, uint32_t expected_rep) {
  float a = fromRep32(a_rep), b = fromRep32(b_rep);
  float x = __addsf3(a, b);
  int ret = compareResultF(x, expected_rep);

  if (ret) {
    printf("error in test__addsf3(%08" PRIx32 ", %08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           a_rep, b_rep, toRep32(x), expected_rep);
  }

  return ret;
}

int main(void) {
  int status = 0;

  // zero + zero = zero
  status |= test__addsf3(0x00000000, 0x00000000, 0x00000000);
  // -zero + -zero = -zero
  status |= test__addsf3(0x80000000, 0x80000000, 0x80000000);
  // zero + -zero = zero
  status |= test__addsf3(0x00000000, 0x80000000, 0x00000000);

  // zero + normal = normal
  status |= test__addsf3(0x00000000, 0x3f800000, 0x3f800000); // 0 + 1.0 = 1.0
  // normal + zero = normal
  status |= test__addsf3(0x3f800000, 0x00000000, 0x3f800000); // 1.0 + 0 = 1.0

  // normal + normal = normal
  status |= test__addsf3(0x3f800000, 0x3f800000, 0x40000000); // 1.0 + 1.0 = 2.0
  status |= test__addsf3(0x40000000, 0x3f800000, 0x40400000); // 2.0 + 1.0 = 3.0
  status |=
      test__addsf3(0x3f800000, 0xbf800000, 0x00000000); // 1.0 + -1.0 = 0.0
  status |=
      test__addsf3(0xbf800000, 0x3f800000, 0x00000000); // -1.0 + 1.0 = 0.0

  // inf + inf = inf
  status |= test__addsf3(0x7f800000, 0x7f800000, 0x7f800000);
  // -inf + -inf = -inf
  status |= test__addsf3(0xff800000, 0xff800000, 0xff800000);
  // inf + -inf = NaN
  status |= test__addsf3(0x7f800000, 0xff800000, 0x7fc00000);
  // inf + normal = inf
  status |= test__addsf3(0x7f800000, 0x3f800000, 0x7f800000);

  // NaN + anything = NaN
  status |= test__addsf3(0x7fc00000, 0x3f800000, 0x7fc00000);
  // anything + NaN = NaN
  status |= test__addsf3(0x3f800000, 0x7fc00000, 0x7fc00000);

  // smallest subnormal + smallest subnormal = 2 * smallest subnormal
  status |= test__addsf3(0x00000001, 0x00000001, 0x00000002);

  // subnormal + subnormal, result stays subnormal
  status |= test__addsf3(0x00000003, 0x00000005, 0x00000008);

  // Denormal addition overflow: two subnormals whose sum is normal.
  // This is the bug case from llvm/llvm-project#185245.
  //
  // 0x004d8ad0 = 0x1.362b4p-127 (subnormal)
  // 0x009b15a0 = 0x1.362b4p-126 (normal, smallest biased exponent)
  status |= test__addsf3(0x004d8ad0, 0x004d8ad0, 0x009b15a0);

  // Half of smallest normal doubled = smallest normal
  // 0x00400000 = 0x1.0p-127 (subnormal), doubled = 0x00800000 = 0x1.0p-126
  status |= test__addsf3(0x00400000, 0x00400000, 0x00800000);

  // Largest subnormal doubled overflows to normal
  // 0x007fffff = largest subnormal, doubled = 0x00fffffe (normal)
  status |= test__addsf3(0x007fffff, 0x007fffff, 0x00fffffe);

  // Negative subnormal overflow to normal
  status |= test__addsf3(0x804d8ad0, 0x804d8ad0, 0x809b15a0);

  // Large normal values near overflow
  status |= test__addsf3(0x7f7fffff, 0x7f7fffff, 0x7f800000); // max + max = inf

  return status;
}
