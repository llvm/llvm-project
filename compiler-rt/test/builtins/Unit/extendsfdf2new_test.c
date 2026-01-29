// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendsfdf2

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// By default this test uses compareResultD to check the returned floats, which
// accepts any returned NaN if the expected result is the canonical NaN value
// 0x7ff8000000000000. For the Arm optimized FP implementation, which commits
// to a more detailed handling of NaNs, we tighten up the check and include
// some extra test cases specific to that NaN policy.
#if (__arm__ && !(__thumb__ && !__thumb2__)) && COMPILER_RT_ARM_OPTIMIZED_FP
#  define EXPECT_EXACT_RESULTS
#  define ARM_NAN_HANDLING
#endif

// Returns: a converted from float to double
COMPILER_RT_ABI double __extendsfdf2(float a);

int test__extendsfdf2(int line, uint32_t a_rep, uint64_t expected_rep) {
  float a = fromRep32(a_rep);
  double x = __extendsfdf2(a);
#ifdef EXPECT_EXACT_RESULTS
  int ret = toRep64(x) != expected_rep;
#else
  int ret = compareResultD(x, expected_rep);
#endif

  if (ret) {
    printf("error at line %d: __extendsfdf2(%08" PRIx32 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a_rep, toRep64(x), expected_rep);
  }
  return ret;
}

#define test__extendsfdf2(a,x) test__extendsfdf2(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__extendsfdf2(0x00000001, 0x36a0000000000000);
  status |= test__extendsfdf2(0x00000003, 0x36b8000000000000);
  status |= test__extendsfdf2(0x00000005, 0x36c4000000000000);
  status |= test__extendsfdf2(0x00000009, 0x36d2000000000000);
  status |= test__extendsfdf2(0x00000011, 0x36e1000000000000);
  status |= test__extendsfdf2(0x00000021, 0x36f0800000000000);
  status |= test__extendsfdf2(0x00000041, 0x3700400000000000);
  status |= test__extendsfdf2(0x00000081, 0x3710200000000000);
  status |= test__extendsfdf2(0x00000101, 0x3720100000000000);
  status |= test__extendsfdf2(0x00000201, 0x3730080000000000);
  status |= test__extendsfdf2(0x00000401, 0x3740040000000000);
  status |= test__extendsfdf2(0x00000801, 0x3750020000000000);
  status |= test__extendsfdf2(0x00001001, 0x3760010000000000);
  status |= test__extendsfdf2(0x00002001, 0x3770008000000000);
  status |= test__extendsfdf2(0x00004001, 0x3780004000000000);
  status |= test__extendsfdf2(0x00008001, 0x3790002000000000);
  status |= test__extendsfdf2(0x00010001, 0x37a0001000000000);
  status |= test__extendsfdf2(0x00020001, 0x37b0000800000000);
  status |= test__extendsfdf2(0x00040001, 0x37c0000400000000);
  status |= test__extendsfdf2(0x00080001, 0x37d0000200000000);
  status |= test__extendsfdf2(0x00100001, 0x37e0000100000000);
  status |= test__extendsfdf2(0x00200001, 0x37f0000080000000);
  status |= test__extendsfdf2(0x00400001, 0x3800000040000000);
  status |= test__extendsfdf2(0x00800001, 0x3810000020000000);
  status |= test__extendsfdf2(0x01000001, 0x3820000020000000);
  status |= test__extendsfdf2(0x20000001, 0x3c00000020000000);
  status |= test__extendsfdf2(0x30000001, 0x3e00000020000000);
  status |= test__extendsfdf2(0x3f800000, 0x3ff0000000000000);
  status |= test__extendsfdf2(0x7f000000, 0x47e0000000000000);
  status |= test__extendsfdf2(0x7f7fffff, 0x47efffffe0000000);
  status |= test__extendsfdf2(0x7f800000, 0x7ff0000000000000);
  status |= test__extendsfdf2(0xff000000, 0xc7e0000000000000);
  status |= test__extendsfdf2(0xff7fffff, 0xc7efffffe0000000);
  status |= test__extendsfdf2(0xff800000, 0xfff0000000000000);
  status |= test__extendsfdf2(0x80800000, 0xb810000000000000);
  status |= test__extendsfdf2(0x807fffff, 0xb80fffffc0000000);
  status |= test__extendsfdf2(0x80400000, 0xb800000000000000);
  status |= test__extendsfdf2(0x803fffff, 0xb7ffffff80000000);
  status |= test__extendsfdf2(0x80000003, 0xb6b8000000000000);
  status |= test__extendsfdf2(0x80000002, 0xb6b0000000000000);
  status |= test__extendsfdf2(0x80000001, 0xb6a0000000000000);
  status |= test__extendsfdf2(0x80000000, 0x8000000000000000);

  // Test that the result of an operation is a NaN at all when it should be.
  //
  // In most configurations these tests' results are checked compared using
  // compareResultD, so we set all the answers to the canonical NaN
  // 0x7ff8000000000000, which causes compareResultF to accept any NaN
  // encoding. We also use the same value as the input NaN in tests that have
  // one, so that even in EXPECT_EXACT_RESULTS mode these tests should pass,
  // because 0x7ff8000000000000 is still the exact expected NaN.
  status |= test__extendsfdf2(0x7fc00000, 0x7ff8000000000000);

#ifdef ARM_NAN_HANDLING
  // Tests specific to the NaN handling of Arm hardware, mimicked by
  // arm/extendsfdf2.S:
  //
  //  - a quiet NaN is distinguished by the top mantissa bit being 1
  //
  //  - converting a quiet NaN from float to double is done by copying
  //    the input mantissa bits to the top of the output mantissa and
  //    appending 0 bits below them
  //
  //  - if the input is a signalling NaN, its top mantissa bit is set
  //    to turn it quiet, and then that quiet NaN is converted to
  //    double as above
  status |= test__extendsfdf2(0x7faf53b1, 0x7ffdea7620000000);
  status |= test__extendsfdf2(0x7fe111d3, 0x7ffc223a60000000);
  status |= test__extendsfdf2(0xffaf53b1, 0xfffdea7620000000);
  status |= test__extendsfdf2(0xffe111d3, 0xfffc223a60000000);

#endif // ARM_NAN_HANDLING

  return status;
}
