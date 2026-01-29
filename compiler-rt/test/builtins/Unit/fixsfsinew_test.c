// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixsfsi

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

// Returns: a converted from float to int32_t
COMPILER_RT_ABI int32_t __fixsfsi(float a);

int test__fixsfsi(int line, uint32_t a_rep, uint32_t expected) {
  float a = fromRep32(a_rep);
  uint32_t x = (uint32_t)__fixsfsi(a);
  int ret = x != expected;

  if (ret) {
    printf("error at line %d: __fixsfsi(%08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a_rep, x, expected);
  }
  return ret;
}

#define test__fixsfsi(a,x) test__fixsfsi(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__fixsfsi(0x00000000, 0x00000000);
  status |= test__fixsfsi(0x00000001, 0x00000000);
  status |= test__fixsfsi(0x00000001, 0x00000000);
  status |= test__fixsfsi(0x00500000, 0x00000000);
  status |= test__fixsfsi(0x00500000, 0x00000000);
  status |= test__fixsfsi(0x3e800000, 0x00000000);
  status |= test__fixsfsi(0x3f000000, 0x00000000);
  status |= test__fixsfsi(0x3f400000, 0x00000000);
  status |= test__fixsfsi(0x3f800000, 0x00000001);
  status |= test__fixsfsi(0x3fa00000, 0x00000001);
  status |= test__fixsfsi(0x3fc00000, 0x00000001);
  status |= test__fixsfsi(0x3fe00000, 0x00000001);
  status |= test__fixsfsi(0x40000000, 0x00000002);
  status |= test__fixsfsi(0x40100000, 0x00000002);
  status |= test__fixsfsi(0x40200000, 0x00000002);
  status |= test__fixsfsi(0x40300000, 0x00000002);
  status |= test__fixsfsi(0x4effffff, 0x7fffff80);
  status |= test__fixsfsi(0x80000000, 0x00000000);
  status |= test__fixsfsi(0x80000001, 0x00000000);
  status |= test__fixsfsi(0x80000001, 0x00000000);
  status |= test__fixsfsi(0x80500000, 0x00000000);
  status |= test__fixsfsi(0x80500000, 0x00000000);
  status |= test__fixsfsi(0xbe800000, 0x00000000);
  status |= test__fixsfsi(0xbf000000, 0x00000000);
  status |= test__fixsfsi(0xbf400000, 0x00000000);
  status |= test__fixsfsi(0xbf800000, 0xffffffff);
  status |= test__fixsfsi(0xbfa00000, 0xffffffff);
  status |= test__fixsfsi(0xbfc00000, 0xffffffff);
  status |= test__fixsfsi(0xbfe00000, 0xffffffff);
  status |= test__fixsfsi(0xc0000000, 0xfffffffe);
  status |= test__fixsfsi(0xc0100000, 0xfffffffe);
  status |= test__fixsfsi(0xc0200000, 0xfffffffe);
  status |= test__fixsfsi(0xc0300000, 0xfffffffe);
  status |= test__fixsfsi(0xcf000000, 0x80000000);

#ifdef ARM_INVALID_HANDLING
  // Tests specific to the handling of float-to-integer conversions in
  // Arm hardware, mimicked by arm/fixsfsi.S:
  //
  //  - too-large positive inputs, including +infinity, return the
  //    maximum possible signed integer value
  //
  //  - too-large negative inputs, including -infinity, return the
  //    minimum possible signed integer value
  //
  //  - NaN inputs return 0
  status |= test__fixsfsi(0x4f000000, 0x7fffffff);
  status |= test__fixsfsi(0x7f800000, 0x7fffffff);
  status |= test__fixsfsi(0x7fa111d3, 0x00000000);
  status |= test__fixsfsi(0x7febfdda, 0x00000000);
  status |= test__fixsfsi(0xcf000001, 0x80000000);
  status |= test__fixsfsi(0xff800000, 0x80000000);

#endif // ARM_INVALID_HANDLING

  return status;
}
