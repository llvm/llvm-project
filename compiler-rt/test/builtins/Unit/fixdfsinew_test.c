// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixdfsi

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

// Returns: a converted from double to int32_t
COMPILER_RT_ABI int32_t __fixdfsi(double a);

int test__fixdfsi(int line, uint64_t a_rep, uint32_t expected) {
  double a = fromRep64(a_rep);
  uint32_t x = (uint32_t)__fixdfsi(a);
  int ret = x != expected;

  if (ret) {
    printf("error at line %d: __fixdfsi(%016" PRIx64 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a_rep, x, expected);
  }
  return ret;
}

#define test__fixdfsi(a,x) test__fixdfsi(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__fixdfsi(0x0000000000000000, 0x00000000);
  status |= test__fixdfsi(0x0000000000000001, 0x00000000);
  status |= test__fixdfsi(0x380a000000000000, 0x00000000);
  status |= test__fixdfsi(0x3fd0000000000000, 0x00000000);
  status |= test__fixdfsi(0x3fe0000000000000, 0x00000000);
  status |= test__fixdfsi(0x3fe8000000000000, 0x00000000);
  status |= test__fixdfsi(0x3ff0000000000000, 0x00000001);
  status |= test__fixdfsi(0x3ff4000000000000, 0x00000001);
  status |= test__fixdfsi(0x3ff8000000000000, 0x00000001);
  status |= test__fixdfsi(0x3ffc000000000000, 0x00000001);
  status |= test__fixdfsi(0x4000000000000000, 0x00000002);
  status |= test__fixdfsi(0x4002000000000000, 0x00000002);
  status |= test__fixdfsi(0x4004000000000000, 0x00000002);
  status |= test__fixdfsi(0x4006000000000000, 0x00000002);
  status |= test__fixdfsi(0x8000000000000000, 0x00000000);
  status |= test__fixdfsi(0x8000000000000001, 0x00000000);
  status |= test__fixdfsi(0xb80a000000000000, 0x00000000);
  status |= test__fixdfsi(0xbfd0000000000000, 0x00000000);
  status |= test__fixdfsi(0xbfe0000000000000, 0x00000000);
  status |= test__fixdfsi(0xbfe8000000000000, 0x00000000);
  status |= test__fixdfsi(0xbff0000000000000, 0xffffffff);
  status |= test__fixdfsi(0xbff4000000000000, 0xffffffff);
  status |= test__fixdfsi(0xbff8000000000000, 0xffffffff);
  status |= test__fixdfsi(0xbffc000000000000, 0xffffffff);
  status |= test__fixdfsi(0xc000000000000000, 0xfffffffe);
  status |= test__fixdfsi(0xc002000000000000, 0xfffffffe);
  status |= test__fixdfsi(0xc004000000000000, 0xfffffffe);
  status |= test__fixdfsi(0xc006000000000000, 0xfffffffe);
  status |= test__fixdfsi(0xc1e0000000000000, 0x80000000);

#ifdef ARM_INVALID_HANDLING
  // Tests specific to the handling of float-to-integer conversions in
  // Arm hardware, mimicked by arm/fixdfsi.S:
  //
  //  - too-large positive inputs, including +infinity, return the
  //    maximum possible signed integer value
  //
  //  - too-large negative inputs, including -infinity, return the
  //    minimum possible signed integer value
  //
  //  - NaN inputs return 0
  status |= test__fixdfsi(0x41dfffffffffffff, 0x7fffffff);
  status |= test__fixdfsi(0x41e0000000000000, 0x7fffffff);
  status |= test__fixdfsi(0x7ff0000000000000, 0x7fffffff);
  status |= test__fixdfsi(0x7ff6d1ebdfe15ee3, 0x00000000);
  status |= test__fixdfsi(0x7ff9a4da74944a09, 0x00000000);
  status |= test__fixdfsi(0xc1e00000001fffff, 0x80000000);
  status |= test__fixdfsi(0xc1e0000000200000, 0x80000000);
  status |= test__fixdfsi(0xfff0000000000000, 0x80000000);

#endif // ARM_INVALID_HANDLING

  return status;
}
