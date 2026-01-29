// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunsdfdi

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// By default this test does not specify the expected results for overflowing
// and NaN inputs, because they can vary between platforms. For the Arm
// optimized FP implementation, which commits to more detail, we include some
// extra test cases specific to that NaN policy.
#if (__arm__ && !(__thumb__ && !__thumb2__)) && COMPILER_RT_ARM_OPTIMIZED_FP
#define ARM_INVALID_HANDLING
#endif

// Returns: a converted from double to int64_t
COMPILER_RT_ABI int64_t __fixunsdfdi(double a);

int test__fixunsdfdi(int line, uint64_t a_rep, uint64_t expected) {
  double a = fromRep64(a_rep);
  uint64_t x = (uint64_t)__fixunsdfdi(a);
  int ret = x != expected;

  if (ret) {
    printf("error at line %d: __fixunsdfdi(%016" PRIx64 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a_rep, x, expected);
  }
  return ret;
}

#define test__fixunsdfdi(a,x) test__fixunsdfdi(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__fixunsdfdi(0x0000000000000000, 0x0000000000000000);
  status |= test__fixunsdfdi(0x0000000000000001, 0x0000000000000000);
  status |= test__fixunsdfdi(0x0000000000500000, 0x0000000000000000);
  status |= test__fixunsdfdi(0x3fd0000000000000, 0x0000000000000000);
  status |= test__fixunsdfdi(0x3fe0000000000000, 0x0000000000000000);
  status |= test__fixunsdfdi(0x3fe8000000000000, 0x0000000000000000);
  status |= test__fixunsdfdi(0x3ff0000000000000, 0x0000000000000001);
  status |= test__fixunsdfdi(0x3ff4000000000000, 0x0000000000000001);
  status |= test__fixunsdfdi(0x3ff8000000000000, 0x0000000000000001);
  status |= test__fixunsdfdi(0x3ffc000000000000, 0x0000000000000001);
  status |= test__fixunsdfdi(0x4000000000000000, 0x0000000000000002);
  status |= test__fixunsdfdi(0x4002000000000000, 0x0000000000000002);
  status |= test__fixunsdfdi(0x4004000000000000, 0x0000000000000002);
  status |= test__fixunsdfdi(0x4006000000000000, 0x0000000000000002);
  status |= test__fixunsdfdi(0x41f0000000040000, 0x0000000100000000);
  status |= test__fixunsdfdi(0x41f0000000080000, 0x0000000100000000);
  status |= test__fixunsdfdi(0x41f00000000c0000, 0x0000000100000000);
  status |= test__fixunsdfdi(0x41f0000000140000, 0x0000000100000001);
  status |= test__fixunsdfdi(0x41f0000000180000, 0x0000000100000001);
  status |= test__fixunsdfdi(0x41f00000001c0000, 0x0000000100000001);
  status |= test__fixunsdfdi(0x41f0000000240000, 0x0000000100000002);
  status |= test__fixunsdfdi(0x41f0000000280000, 0x0000000100000002);
  status |= test__fixunsdfdi(0x41f00000002c0000, 0x0000000100000002);
  status |= test__fixunsdfdi(0x41fffffffff40000, 0x00000001ffffffff);
  status |= test__fixunsdfdi(0x41fffffffff80000, 0x00000001ffffffff);
  status |= test__fixunsdfdi(0x41fffffffffc0000, 0x00000001ffffffff);
  status |= test__fixunsdfdi(0x42a0468ace000000, 0x0000082345670000);
  status |= test__fixunsdfdi(0x43efffffffffffff, 0xfffffffffffff800);
  status |= test__fixunsdfdi(0x8000000000000000, 0x0000000000000000);

#ifdef ARM_INVALID_HANDLING
  // Tests specific to the handling of float-to-integer conversions in
  // Arm hardware, mimicked by arm/fixunsdfsi.S:
  //
  //  - too-large positive inputs, including +infinity, return the
  //    maximum possible unsigned integer value
  //
  //  - negative inputs too small to round up to 0, including
  //    -infinity, return 0
  //
  //  - NaN inputs return 0
  status |= test__fixunsdfdi(0x43f0000000000000, 0xffffffffffffffff);
  status |= test__fixunsdfdi(0x7ff0000000000000, 0xffffffffffffffff);
  status |= test__fixunsdfdi(0x7ff6d1ebdfe15ee3, 0x0000000000000000);
  status |= test__fixunsdfdi(0x7ff9a4da74944a09, 0x0000000000000000);
  status |= test__fixunsdfdi(0xbfefffffffffffff, 0x0000000000000000);
  status |= test__fixunsdfdi(0xbff0000000000000, 0x0000000000000000);
  status |= test__fixunsdfdi(0xc000000000000000, 0x0000000000000000);
  status |= test__fixunsdfdi(0xfff0000000000000, 0x0000000000000000);

#endif // ARM_INVALID_HANDLING

  return status;
}
