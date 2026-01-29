// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixdfdi

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
COMPILER_RT_ABI int64_t __fixdfdi(double a);

int test__fixdfdi(int line, uint64_t a_rep, uint64_t expected) {
  double a = fromRep64(a_rep);
  uint64_t x = (uint64_t)__fixdfdi(a);
  int ret = x != expected;

  if (ret) {
    printf("error at line %d: __fixdfdi(%016" PRIx64 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a_rep, x, expected);
  }
  return ret;
}

#define test__fixdfdi(a,x) test__fixdfdi(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__fixdfdi(0x0000000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0x0000000000000001, 0x0000000000000000);
  status |= test__fixdfdi(0x0000000000500000, 0x0000000000000000);
  status |= test__fixdfdi(0x3fd0000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0x3fe0000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0x3fe8000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0x3ff0000000000000, 0x0000000000000001);
  status |= test__fixdfdi(0x3ff4000000000000, 0x0000000000000001);
  status |= test__fixdfdi(0x3ff8000000000000, 0x0000000000000001);
  status |= test__fixdfdi(0x3ffc000000000000, 0x0000000000000001);
  status |= test__fixdfdi(0x4000000000000000, 0x0000000000000002);
  status |= test__fixdfdi(0x4002000000000000, 0x0000000000000002);
  status |= test__fixdfdi(0x4004000000000000, 0x0000000000000002);
  status |= test__fixdfdi(0x4006000000000000, 0x0000000000000002);
  status |= test__fixdfdi(0x41f0000000040000, 0x0000000100000000);
  status |= test__fixdfdi(0x41f0000000080000, 0x0000000100000000);
  status |= test__fixdfdi(0x41f00000000c0000, 0x0000000100000000);
  status |= test__fixdfdi(0x41f0000000140000, 0x0000000100000001);
  status |= test__fixdfdi(0x41f0000000180000, 0x0000000100000001);
  status |= test__fixdfdi(0x41f00000001c0000, 0x0000000100000001);
  status |= test__fixdfdi(0x41f0000000240000, 0x0000000100000002);
  status |= test__fixdfdi(0x41f0000000280000, 0x0000000100000002);
  status |= test__fixdfdi(0x41f00000002c0000, 0x0000000100000002);
  status |= test__fixdfdi(0x41fffffffff40000, 0x00000001ffffffff);
  status |= test__fixdfdi(0x41fffffffff80000, 0x00000001ffffffff);
  status |= test__fixdfdi(0x41fffffffffc0000, 0x00000001ffffffff);
  status |= test__fixdfdi(0x42a0468ace000000, 0x0000082345670000);
  status |= test__fixdfdi(0x43dfffffffffffff, 0x7ffffffffffffc00);
  status |= test__fixdfdi(0x8000000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0x8000000000000001, 0x0000000000000000);
  status |= test__fixdfdi(0x8000000000500000, 0x0000000000000000);
  status |= test__fixdfdi(0xbfd0000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0xbfe0000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0xbfe8000000000000, 0x0000000000000000);
  status |= test__fixdfdi(0xbff0000000000000, 0xffffffffffffffff);
  status |= test__fixdfdi(0xbff4000000000000, 0xffffffffffffffff);
  status |= test__fixdfdi(0xbff8000000000000, 0xffffffffffffffff);
  status |= test__fixdfdi(0xbffc000000000000, 0xffffffffffffffff);
  status |= test__fixdfdi(0xc000000000000000, 0xfffffffffffffffe);
  status |= test__fixdfdi(0xc002000000000000, 0xfffffffffffffffe);
  status |= test__fixdfdi(0xc004000000000000, 0xfffffffffffffffe);
  status |= test__fixdfdi(0xc006000000000000, 0xfffffffffffffffe);
  status |= test__fixdfdi(0xc1f0000000040000, 0xffffffff00000000);
  status |= test__fixdfdi(0xc1f0000000080000, 0xffffffff00000000);
  status |= test__fixdfdi(0xc1f00000000c0000, 0xffffffff00000000);
  status |= test__fixdfdi(0xc1f0000000140000, 0xfffffffeffffffff);
  status |= test__fixdfdi(0xc1f0000000180000, 0xfffffffeffffffff);
  status |= test__fixdfdi(0xc1f00000001c0000, 0xfffffffeffffffff);
  status |= test__fixdfdi(0xc1f0000000240000, 0xfffffffefffffffe);
  status |= test__fixdfdi(0xc1f0000000280000, 0xfffffffefffffffe);
  status |= test__fixdfdi(0xc1f00000002c0000, 0xfffffffefffffffe);
  status |= test__fixdfdi(0xc1fffffffff40000, 0xfffffffe00000001);
  status |= test__fixdfdi(0xc1fffffffff80000, 0xfffffffe00000001);
  status |= test__fixdfdi(0xc1fffffffffc0000, 0xfffffffe00000001);
  status |= test__fixdfdi(0xc3dfffffffffffff, 0x8000000000000400);
  status |= test__fixdfdi(0xc3e0000000000000, 0x8000000000000000);

#ifdef ARM_INVALID_HANDLING
  // Tests specific to the handling of float-to-integer conversions in
  // Arm hardware, mimicked by arm/fixdfdi.S:
  //
  //  - too-large positive inputs, including +infinity, return the
  //    maximum possible signed integer value
  //
  //  - too-large negative inputs, including -infinity, return the
  //    minimum possible signed integer value
  //
  //  - NaN inputs return 0
  status |= test__fixdfdi(0x43e0000000000000, 0x7fffffffffffffff);
  status |= test__fixdfdi(0x7ff0000000000000, 0x7fffffffffffffff);
  status |= test__fixdfdi(0x7ff6d1ebdfe15ee3, 0x0000000000000000);
  status |= test__fixdfdi(0x7ff9a4da74944a09, 0x0000000000000000);
  status |= test__fixdfdi(0xc3e0000000000001, 0x8000000000000000);
  status |= test__fixdfdi(0xfff0000000000000, 0x8000000000000000);

#endif // ARM_INVALID_HANDLING

  return status;
}
