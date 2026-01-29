// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunssfsi

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

// Returns: a converted from float to uint32_t
COMPILER_RT_ABI uint32_t __fixunssfsi(float a);

int test__fixunssfsi(int line, uint32_t a_rep, uint32_t expected) {
  float a = fromRep32(a_rep);
  int32_t x = __fixunssfsi(a);
  int ret = x != expected;

  if (ret) {
    printf("error at line %d: __fixunssfsi(%08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a_rep, x, expected);
  }
  return ret;
}

#define test__fixunssfsi(a,x) test__fixunssfsi(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__fixunssfsi(0x00000000, 0x00000000);
  status |= test__fixunssfsi(0x00000001, 0x00000000);
  status |= test__fixunssfsi(0x00000001, 0x00000000);
  status |= test__fixunssfsi(0x00500000, 0x00000000);
  status |= test__fixunssfsi(0x00500000, 0x00000000);
  status |= test__fixunssfsi(0x3e800000, 0x00000000);
  status |= test__fixunssfsi(0x3f000000, 0x00000000);
  status |= test__fixunssfsi(0x3f400000, 0x00000000);
  status |= test__fixunssfsi(0x3f800000, 0x00000001);
  status |= test__fixunssfsi(0x3fa00000, 0x00000001);
  status |= test__fixunssfsi(0x3fc00000, 0x00000001);
  status |= test__fixunssfsi(0x3fe00000, 0x00000001);
  status |= test__fixunssfsi(0x40000000, 0x00000002);
  status |= test__fixunssfsi(0x40100000, 0x00000002);
  status |= test__fixunssfsi(0x40200000, 0x00000002);
  status |= test__fixunssfsi(0x40300000, 0x00000002);
  status |= test__fixunssfsi(0x4f7fffff, 0xffffff00);
  status |= test__fixunssfsi(0x80000000, 0x00000000);
  status |= test__fixunssfsi(0xbf7fffff, 0x00000000);

#ifdef ARM_INVALID_HANDLING
  // Tests specific to the handling of float-to-integer conversions in
  // Arm hardware, mimicked by arm/fixunssfsi.S:
  //
  //  - too-large positive inputs, including +infinity, return the
  //    maximum possible unsigned integer value
  //
  //  - negative inputs too small to round up to 0, including
  //    -infinity, return 0
  //
  //  - NaN inputs return 0
  status |= test__fixunssfsi(0x4f800000, 0xffffffff);
  status |= test__fixunssfsi(0x7f800000, 0xffffffff);
  status |= test__fixunssfsi(0x7fa111d3, 0x00000000);
  status |= test__fixunssfsi(0x7febfdda, 0x00000000);
  status |= test__fixunssfsi(0xbf800000, 0x00000000);
  status |= test__fixunssfsi(0xc0000000, 0x00000000);
  status |= test__fixunssfsi(0xff800000, 0x00000000);

#endif // ARM_INVALID_HANDLING

  return status;
}
