// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divdf3

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

// Returns: a / b
COMPILER_RT_ABI double __divdf3(double a, double b);

int test__divdf3(int line, uint64_t a_rep, uint64_t b_rep, uint64_t expected_rep) {
  double a = fromRep64(a_rep), b = fromRep64(b_rep);
  double x = __divdf3(a, b);
#ifdef EXPECT_EXACT_RESULTS
  int ret = toRep64(x) != expected_rep;
#else
  int ret = compareResultD(x, expected_rep);
#endif

  if (ret) {
    printf("error at line %d: __divdf3(%016" PRIx64 ", %016" PRIx64 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a_rep, b_rep, toRep64(x), expected_rep);
  }
  return ret;
}

#define test__divdf3(a,b,x) test__divdf3(__LINE__,a,b,x)

int main(void) {
  int status = 0;

  status |= test__divdf3(0x0000000000000000, 0x0000000000000001, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x000fffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x0010000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x001fffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x3ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x4014000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x7fdfffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x7fe0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x8000000000000002, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x800fffffffffffff, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x8010000000000001, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0x8020000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0xc008000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0xc01c000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0xffcfffffffffffff, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0xffe0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000000, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000001, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x0000000000000001, 0x3fc0000000000000, 0x0000000000000008);
  status |= test__divdf3(0x0000000000000001, 0x3fe0000000000000, 0x0000000000000002);
  status |= test__divdf3(0x0000000000000001, 0x4000000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000001, 0x7fefffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000001, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0000000000000001, 0xc000000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000001, 0xffefffffffffffff, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000002, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x0000000000000002, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0000000000000009, 0x4022000000000000, 0x0000000000000001);
  status |= test__divdf3(0x0000000000000009, 0xc022000000000000, 0x8000000000000001);
  status |= test__divdf3(0x000ffffffffffff7, 0x3feffffffffffffe, 0x000ffffffffffff8);
  status |= test__divdf3(0x000ffffffffffffe, 0x3feffffffffffffe, 0x000fffffffffffff);
  status |= test__divdf3(0x000fffffffffffff, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x000fffffffffffff, 0x3f60000000000000, 0x009ffffffffffffe);
  status |= test__divdf3(0x000fffffffffffff, 0x3fe0000000000000, 0x001ffffffffffffe);
  status |= test__divdf3(0x000fffffffffffff, 0x3ff0000000000000, 0x000fffffffffffff);
  status |= test__divdf3(0x000fffffffffffff, 0x3ff0000000000002, 0x000ffffffffffffd);
  status |= test__divdf3(0x000fffffffffffff, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x000fffffffffffff, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x000fffffffffffff, 0xbff0000000000000, 0x800fffffffffffff);
  status |= test__divdf3(0x000fffffffffffff, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0010000000000000, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x0010000000000000, 0x3ff0000000000001, 0x000fffffffffffff);
  status |= test__divdf3(0x0010000000000000, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0010000000000001, 0x3ff0000000000002, 0x000fffffffffffff);
  status |= test__divdf3(0x0010000000000001, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x0010000000000001, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0010000000000002, 0x3ff0000000000006, 0x000ffffffffffffc);
  status |= test__divdf3(0x001ffffffffffffe, 0x4000000000000000, 0x000fffffffffffff);
  status |= test__divdf3(0x001fffffffffffff, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x001fffffffffffff, 0x4000000000000000, 0x0010000000000000);
  status |= test__divdf3(0x001fffffffffffff, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x0020000000000000, 0x0010000000000000, 0x4000000000000000);
  status |= test__divdf3(0x0020000000000000, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x0020000000000000, 0xc000000000000000, 0x8010000000000000);
  status |= test__divdf3(0x0020000000000000, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x0020000000000001, 0x0010000000000001, 0x4000000000000000);
  status |= test__divdf3(0x0020000000000001, 0xc000000000000000, 0x8010000000000001);
  status |= test__divdf3(0x0020000000000003, 0x8010000000000003, 0xc000000000000000);
  status |= test__divdf3(0x0020000000000003, 0xc000000000000000, 0x8010000000000003);
  status |= test__divdf3(0x3feffffffffffff7, 0x3feffffffffffffb, 0x3feffffffffffffc);
  status |= test__divdf3(0x3feffffffffffff7, 0x3feffffffffffffe, 0x3feffffffffffff9);
  status |= test__divdf3(0x3feffffffffffff8, 0x3feffffffffffffc, 0x3feffffffffffffc);
  status |= test__divdf3(0x3feffffffffffff8, 0x3feffffffffffffd, 0x3feffffffffffffb);
  status |= test__divdf3(0x3feffffffffffffa, 0x3feffffffffffff9, 0x3ff0000000000001);
  status |= test__divdf3(0x3feffffffffffffb, 0x3feffffffffffff9, 0x3ff0000000000001);
  status |= test__divdf3(0x3feffffffffffffc, 0x3feffffffffffff9, 0x3ff0000000000002);
  status |= test__divdf3(0x3feffffffffffffc, 0x3feffffffffffffd, 0x3fefffffffffffff);
  status |= test__divdf3(0x3feffffffffffffc, 0x3feffffffffffffe, 0x3feffffffffffffe);
  status |= test__divdf3(0x3feffffffffffffc, 0x3fefffffffffffff, 0x3feffffffffffffd);
  status |= test__divdf3(0x3feffffffffffffc, 0x3ff0000000000001, 0x3feffffffffffffa);
  status |= test__divdf3(0x3feffffffffffffd, 0x3feffffffffffff9, 0x3ff0000000000002);
  status |= test__divdf3(0x3feffffffffffffd, 0x3feffffffffffffc, 0x3ff0000000000001);
  status |= test__divdf3(0x3feffffffffffffd, 0x3feffffffffffffe, 0x3fefffffffffffff);
  status |= test__divdf3(0x3feffffffffffffd, 0x3fefffffffffffff, 0x3feffffffffffffe);
  status |= test__divdf3(0x3feffffffffffffd, 0x3ff0000000000001, 0x3feffffffffffffb);
  status |= test__divdf3(0x3feffffffffffffd, 0x3ff0000000000002, 0x3feffffffffffff9);
  status |= test__divdf3(0x3feffffffffffffe, 0x3feffffffffffff9, 0x3ff0000000000003);
  status |= test__divdf3(0x3feffffffffffffe, 0x3feffffffffffffc, 0x3ff0000000000001);
  status |= test__divdf3(0x3feffffffffffffe, 0x3feffffffffffffd, 0x3ff0000000000001);
  status |= test__divdf3(0x3feffffffffffffe, 0x3fefffffffffffff, 0x3fefffffffffffff);
  status |= test__divdf3(0x3feffffffffffffe, 0x3ff0000000000001, 0x3feffffffffffffc);
  status |= test__divdf3(0x3feffffffffffffe, 0x3ff0000000000002, 0x3feffffffffffffa);
  status |= test__divdf3(0x3feffffffffffffe, 0x3ff0000000000003, 0x3feffffffffffff8);
  status |= test__divdf3(0x3fefffffffffffff, 0x3feffffffffffff9, 0x3ff0000000000003);
  status |= test__divdf3(0x3fefffffffffffff, 0x3feffffffffffffc, 0x3ff0000000000002);
  status |= test__divdf3(0x3fefffffffffffff, 0x3feffffffffffffd, 0x3ff0000000000001);
  status |= test__divdf3(0x3fefffffffffffff, 0x3feffffffffffffe, 0x3ff0000000000001);
  status |= test__divdf3(0x3fefffffffffffff, 0x3ff0000000000001, 0x3feffffffffffffd);
  status |= test__divdf3(0x3fefffffffffffff, 0x3ff0000000000002, 0x3feffffffffffffb);
  status |= test__divdf3(0x3fefffffffffffff, 0x3ff0000000000003, 0x3feffffffffffff9);
  status |= test__divdf3(0x3fefffffffffffff, 0x3ff0000000000004, 0x3feffffffffffff7);
  status |= test__divdf3(0x3ff0000000000000, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x3ff0000000000000, 0x3feffffffffffff7, 0x3ff0000000000005);
  status |= test__divdf3(0x3ff0000000000000, 0x3feffffffffffff8, 0x3ff0000000000004);
  status |= test__divdf3(0x3ff0000000000000, 0x3feffffffffffffb, 0x3ff0000000000003);
  status |= test__divdf3(0x3ff0000000000000, 0x3feffffffffffffc, 0x3ff0000000000002);
  status |= test__divdf3(0x3ff0000000000000, 0x3feffffffffffffd, 0x3ff0000000000002);
  status |= test__divdf3(0x3ff0000000000000, 0x3feffffffffffffe, 0x3ff0000000000001);
  status |= test__divdf3(0x3ff0000000000000, 0x3fefffffffffffff, 0x3ff0000000000001);
  status |= test__divdf3(0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000);
  status |= test__divdf3(0x3ff0000000000000, 0x3ff0000000000001, 0x3feffffffffffffe);
  status |= test__divdf3(0x3ff0000000000000, 0x3ff0000000000002, 0x3feffffffffffffc);
  status |= test__divdf3(0x3ff0000000000000, 0x3ff0000000000003, 0x3feffffffffffffa);
  status |= test__divdf3(0x3ff0000000000000, 0x3ff0000000000004, 0x3feffffffffffff8);
  status |= test__divdf3(0x3ff0000000000000, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x3ff0000000000001, 0x3feffffffffffffb, 0x3ff0000000000004);
  status |= test__divdf3(0x3ff0000000000001, 0x3feffffffffffffd, 0x3ff0000000000003);
  status |= test__divdf3(0x3ff0000000000001, 0x3feffffffffffffe, 0x3ff0000000000002);
  status |= test__divdf3(0x3ff0000000000001, 0x3fefffffffffffff, 0x3ff0000000000002);
  status |= test__divdf3(0x3ff0000000000001, 0x3ff0000000000002, 0x3feffffffffffffe);
  status |= test__divdf3(0x3ff0000000000001, 0x3ff0000000000003, 0x3feffffffffffffc);
  status |= test__divdf3(0x3ff0000000000002, 0x3feffffffffffffc, 0x3ff0000000000004);
  status |= test__divdf3(0x3ff0000000000002, 0x3feffffffffffffd, 0x3ff0000000000004);
  status |= test__divdf3(0x3ff0000000000002, 0x3feffffffffffffe, 0x3ff0000000000003);
  status |= test__divdf3(0x3ff0000000000002, 0x3fefffffffffffff, 0x3ff0000000000003);
  status |= test__divdf3(0x3ff0000000000002, 0x3ff0000000000001, 0x3ff0000000000001);
  status |= test__divdf3(0x3ff0000000000002, 0x3ff0000000000003, 0x3feffffffffffffe);
  status |= test__divdf3(0x3ff0000000000003, 0x3feffffffffffffd, 0x3ff0000000000005);
  status |= test__divdf3(0x3ff0000000000003, 0x3feffffffffffffe, 0x3ff0000000000004);
  status |= test__divdf3(0x3ff0000000000003, 0x3fefffffffffffff, 0x3ff0000000000004);
  status |= test__divdf3(0x3ff0000000000003, 0x3ff0000000000001, 0x3ff0000000000002);
  status |= test__divdf3(0x3ff0000000000004, 0x3feffffffffffffe, 0x3ff0000000000005);
  status |= test__divdf3(0x3ff0000000000004, 0x3ff0000000000001, 0x3ff0000000000003);
  status |= test__divdf3(0x3ff0000000000004, 0x3ff0000000000007, 0x3feffffffffffffa);
  status |= test__divdf3(0x3ff0000000000005, 0x3fefffffffffffff, 0x3ff0000000000006);
  status |= test__divdf3(0x3ff0000000000006, 0x3ff0000000000008, 0x3feffffffffffffc);
  status |= test__divdf3(0x3ff0000000000007, 0x3ff0000000000002, 0x3ff0000000000005);
  status |= test__divdf3(0x3ff0000000000009, 0x3ff0000000000008, 0x3ff0000000000001);
  status |= test__divdf3(0x3ff199999999999a, 0x3ff3333333333333, 0x3fed555555555556);
  status |= test__divdf3(0x4000000000000000, 0x3ff0000000000000, 0x4000000000000000);
  status |= test__divdf3(0x4000000000000000, 0xbff0000000000000, 0xc000000000000000);
  status |= test__divdf3(0x4008000000000000, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x4008000000000000, 0xc008000000000000, 0xbff0000000000000);
  status |= test__divdf3(0x4008000000000000, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x4014000000000000, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x4014000000000000, 0x4014000000000000, 0x3ff0000000000000);
  status |= test__divdf3(0x4014000000000000, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x401c000000000000, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x401c000000000000, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x4020000000000000, 0x4000000000000000, 0x4010000000000000);
  status |= test__divdf3(0x4022000000000000, 0x4008000000000000, 0x4008000000000000);
  status |= test__divdf3(0x7f60000000000000, 0x00a0000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7fcfffffffffffff, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7fdffffffffffffd, 0xc000000000000000, 0xffcffffffffffffd);
  status |= test__divdf3(0x7fdfffffffffffff, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7fdfffffffffffff, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0x000fffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0x3fe0000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0x4000000000000000, 0x7fd0000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0x7ff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0xbfe0000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0xc000000000000000, 0xffd0000000000000);
  status |= test__divdf3(0x7fe0000000000000, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x7fe0000000000003, 0xffd0000000000003, 0xc000000000000000);
  status |= test__divdf3(0x7feffffffffffffd, 0x4010000000000000, 0x7fcffffffffffffd);
  status |= test__divdf3(0x7feffffffffffffd, 0xc010000000000000, 0xffcffffffffffffd);
  status |= test__divdf3(0x7fefffffffffffff, 0x0000000000000001, 0x7ff0000000000000);
  status |= test__divdf3(0x7fefffffffffffff, 0x3fefffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0x7fefffffffffffff, 0x7fcfffffffffffff, 0x4010000000000000);
  status |= test__divdf3(0x7fefffffffffffff, 0x7fdfffffffffffff, 0x4000000000000000);
  status |= test__divdf3(0x7fefffffffffffff, 0xc000000000000000, 0xffdfffffffffffff);
  status |= test__divdf3(0x7fefffffffffffff, 0xffcfffffffffffff, 0xc010000000000000);
  status |= test__divdf3(0x7fefffffffffffff, 0xfff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x0000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x0000000000000001, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x000fffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x0010000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x001fffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x3ff0000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x4014000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x7fdfffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x7fe0000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x8000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x8000000000000002, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x800fffffffffffff, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x8010000000000001, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x8020000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0xc008000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0xc01c000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0xffcfffffffffffff, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0xffe0000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0xffefffffffffffff, 0xfff0000000000000);
  status |= test__divdf3(0x8000000000000000, 0x0000000000000003, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x000fffffffffffff, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x0010000000000001, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x0020000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x4000000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x4018000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x7fcfffffffffffff, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x7fd0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x8000000000000004, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x800fffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x8010000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0x801fffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0xc010000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0xc020000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0xffd0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0xffdfffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000000, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000001, 0x3fe0000000000000, 0x8000000000000002);
  status |= test__divdf3(0x8000000000000001, 0x4000000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000001, 0x7fefffffffffffff, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000001, 0xc000000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000001, 0xffefffffffffffff, 0x0000000000000000);
  status |= test__divdf3(0x8000000000000003, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x8000000000000003, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8000000000000004, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x8000000000000004, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x800ffffffffffff8, 0x3feffffffffffffe, 0x800ffffffffffff9);
  status |= test__divdf3(0x800fffffffffffff, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x800fffffffffffff, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x800fffffffffffff, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x800fffffffffffff, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8010000000000000, 0x3ff0000000000001, 0x800fffffffffffff);
  status |= test__divdf3(0x8010000000000000, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x8010000000000000, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8010000000000001, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x8010000000000001, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x801fffffffffffff, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0x801fffffffffffff, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0x8020000000000000, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0x8020000000000000, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0x8020000000000001, 0x0010000000000001, 0xc000000000000000);
  status |= test__divdf3(0x8020000000000005, 0x0010000000000005, 0xc000000000000000);
  status |= test__divdf3(0xbff0000000000000, 0x3ff0000000000000, 0xbff0000000000000);
  status |= test__divdf3(0xbff0000000000000, 0xbff0000000000000, 0x3ff0000000000000);
  status |= test__divdf3(0xc000000000000000, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xc000000000000000, 0x3ff0000000000000, 0xc000000000000000);
  status |= test__divdf3(0xc000000000000000, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0xc000000000000000, 0xbff0000000000000, 0x4000000000000000);
  status |= test__divdf3(0xc010000000000000, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xc010000000000000, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0xc018000000000000, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xc018000000000000, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0xc018000000000000, 0xc008000000000000, 0x4000000000000000);
  status |= test__divdf3(0xc01c000000000000, 0x401c000000000000, 0xbff0000000000000);
  status |= test__divdf3(0xc020000000000000, 0x4000000000000000, 0xc010000000000000);
  status |= test__divdf3(0xc020000000000000, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xc020000000000000, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0xc022000000000000, 0xc008000000000000, 0x4008000000000000);
  status |= test__divdf3(0xffcfffffffffffff, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xffcfffffffffffff, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0xffd0000000000000, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xffd0000000000000, 0x7ff0000000000000, 0x8000000000000000);
  status |= test__divdf3(0xffd0000000000000, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xffd0000000000000, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0xffdfffffffffffff, 0x4000000000000000, 0xffcfffffffffffff);
  status |= test__divdf3(0xffdfffffffffffff, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xffe0000000000000, 0x3fe0000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xffe0000000000000, 0xbfe0000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xffe0000000000001, 0x7fd0000000000001, 0xc000000000000000);
  status |= test__divdf3(0xffeffffffffffffd, 0x4010000000000000, 0xffcffffffffffffd);
  status |= test__divdf3(0xffeffffffffffffd, 0xc010000000000000, 0x7fcffffffffffffd);
  status |= test__divdf3(0xffefffffffffffff, 0x7fcfffffffffffff, 0xc010000000000000);
  status |= test__divdf3(0xffefffffffffffff, 0xffcfffffffffffff, 0x4010000000000000);
  status |= test__divdf3(0xffefffffffffffff, 0xfff0000000000000, 0x0000000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x0000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x0000000000000003, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x000fffffffffffff, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x0010000000000001, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x0020000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x4000000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x4018000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x7fd0000000000000, 0xfff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x8000000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x8000000000000004, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x800fffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x8010000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x801fffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0xc010000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0xc020000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0xffd0000000000000, 0x7ff0000000000000);
  status |= test__divdf3(0xfff0000000000000, 0xffefffffffffffff, 0x7ff0000000000000);
  status |= test__divdf3(0x800ffffffdffffff, 0xc00fff8000000000, 0x0004000fffbfff00);
  status |= test__divdf3(0xb7fbffffffffffff, 0xffe0000000000007, 0x0000000000000000);
  status |= test__divdf3(0x3ff660beb3029ffd, 0x3ff52e22fb7ace43, 0x3ff0e79e59ccb735);
  status |= test__divdf3(0x3ff73ddbc621eb00, 0x3ffb8224c030d747, 0x3feb095d4073d13b);
  status |= test__divdf3(0x3ff9a3b1ff2bf973, 0x3ff42fdf35d2d3bd, 0x3ff452508f203fca);
  status |= test__divdf3(0x3ffa2f42f2a01655, 0x3ff01310ba9f33d1, 0x3ffa103474220298);
  status |= test__divdf3(0x3ffa6b3e65d68478, 0x3ff773ca580800a9, 0x3ff206204bf651cc);
  status |= test__divdf3(0x3ffae840ed05aaad, 0x3ff374c8afa6bd73, 0x3ff620a0b38357dd);
  status |= test__divdf3(0x3ffc9bff90e124f7, 0x3ff19678d03f31b9, 0x3ffa06ce5731c244);
  status |= test__divdf3(0x3ff716518068f63e, 0x3ffea080001fffff, 0x3fe81f4927e2f813);
  status |= test__divdf3(0x3ff30b70c9e177b3, 0x3ffdc1dbcddeaaf7, 0x3fe47ae453d79b63);
  status |= test__divdf3(0x3ff690a0c1cf289e, 0x3ffdd0e4ec596ead, 0x3fe837c35c721292);
  status |= test__divdf3(0x3ff9a9f18698d1c5, 0x3ffdcf214b672807, 0x3feb8cd196d1e2db);
  status |= test__divdf3(0x3ffc412def95e9f2, 0x3ffe09fd73e44afb, 0x3fee195e4c411819);
  status |= test__divdf3(0x3ffab674f26df917, 0x3ffe55a80dfd623d, 0x3fec2de561fb628a);
  status |= test__divdf3(0x3ff15bb10851a33b, 0x3ffe770229894d4f, 0x3fe23b9bdf3ad4d7);
  status |= test__divdf3(0x3ff6ce035de00c24, 0x3fff04076d288c95, 0x3fe7874738e5ef5e);
  status |= test__divdf3(0x3ffb0e73f83fd2b4, 0x3fff01150ca4f6e3, 0x3febece97e64ff65);
  status |= test__divdf3(0x3ff53fff6c6d7043, 0x3fffb55c0bf15be1, 0x3fe57204f8441410);
  status |= test__divdf3(0x3ffa8aa3bbff7c4b, 0x3fffd530fa74cc5f, 0x3feaae55281a47cf);
  status |= test__divdf3(0x3ff3004b0d901379, 0x3ffe470662686931, 0x3fe41508eef9d818);
  status |= test__divdf3(0x3ffac10f29e80b25, 0x3ffe2fba9d423c9d, 0x3fec5c8a8148eb26);
  status |= test__divdf3(0x3ff8a3e14fe0651f, 0x3ffdeeae50e07679, 0x3fea579ce7a3f61c);
  status |= test__divdf3(0x3ff168321760dd0d, 0x3ffd382a2b3c2c27, 0x3fe31042c5fcbe35);
  status |= test__divdf3(0x3ff208350f930e99, 0x3ffc80beeab6d9ed, 0x3fe43e9486314a0e);
  status |= test__divdf3(0x3ff46a9470b46af6, 0x3ffc2e13c9335b3f, 0x3fe72f150e86f5a1);
  status |= test__divdf3(0x3ffaf26f45d21562, 0x3ffbe6d631b290e7, 0x3feee7b30b353e95);
  status |= test__divdf3(0x3ff5cda6f52381df, 0x3ffbe2a5bce4483f, 0x3fe90542a0e62c21);
  status |= test__divdf3(0x3ff92aeb8209bb69, 0x3ffb57a0bdf7af6f, 0x3fed74754022b839);
  status |= test__divdf3(0x3ff627c9c1a1903d, 0x3ffb3c161457a7e1, 0x3fea082feee891f0);
  status |= test__divdf3(0x3ffa5fef91208fd5, 0x3ff68928392cf5e7, 0x3ff2b9c16cd0a6eb);
  status |= test__divdf3(0x3ffdc6825d6a2ad2, 0x3ff69bb9ca89cd3f, 0x3ff5127c1399515f);
  status |= test__divdf3(0x3ffd62dbb1150699, 0x3ff6e12d3daf7823, 0x3ff48cd52e787bc5);
  status |= test__divdf3(0x3ffb9f0e3f946dd2, 0x3ff75a51f01f688b, 0x3ff2ecadebdfdf91);
  status |= test__divdf3(0x3ffdf21fc13ef609, 0x3ff77a80c8098ae1, 0x3ff46843217c9c90);
  status |= test__divdf3(0x3ff83f6d28924d31, 0x3ff7cb607bcc758f, 0x3ff04e08e26c84b7);
  status |= test__divdf3(0x3ffef8774307cea5, 0x3ff849124d13461d, 0x3ff467851369d61a);
  status |= test__divdf3(0x3ffd7c2259068fa2, 0x3ffa9e9faf8d6845, 0x3ff1b8e24ddeb546);
  status |= test__divdf3(0x3fffb10b35d3977b, 0x3ffb57a0bdf7af6f, 0x3ff28b8abfdd47c7);
  status |= test__divdf3(0x3ffdcfa4097387f1, 0x3ffbe6d631b290e7, 0x3ff1184cf4cac16b);
  status |= test__divdf3(0x3ffcb6231a615d02, 0x3ffb98faef6f9417, 0x3ff0a552a67a8e2d);
  status |= test__divdf3(0x3ffba5443a5d0a42, 0x3ffb3a5c10922a9d, 0x3ff03ed2622d2a26);
  status |= test__divdf3(0x3fff3144ae86b33e, 0x3ffa58948417f235, 0x3ff2f17912d557f2);
  status |= test__divdf3(0x3ffd68635bf6605a, 0x3ff945fce3a79f3f, 0x3ff29e0c7d6617a1);
  status |= test__divdf3(0x3ff97e6030354676, 0x3ff906f78f460697, 0x3ff04c56a5f3136d);
  status |= test__divdf3(0x3ffe86f743594e95, 0x3ff8444d7946422d, 0x3ff420b1e63f512e);
  status |= test__divdf3(0x3fff12a6c5539a9a, 0x3ff7cad48079af09, 0x3ff4e564f736b864);
  status |= test__divdf3(0x3ffa5371fe989251, 0x3ff6fc5272dc36d1, 0x3ff2533d7a4d0ee8);
  status |= test__divdf3(0x3ffe18c0547f65d2, 0x3ff6fc9e8dd915ed, 0x3ff4f2e7f917b80e);
  status |= test__divdf3(0x3ffd7aea8a297055, 0x3ff64eb95d608cd9, 0x3ff52500dc28664c);

  // Test that the result of an operation is a NaN at all when it should be.
  //
  // In most configurations these tests' results are checked compared using
  // compareResultD, so we set all the answers to the canonical NaN
  // 0x7ff8000000000000, which causes compareResultF to accept any NaN
  // encoding. We also use the same value as the input NaN in tests that have
  // one, so that even in EXPECT_EXACT_RESULTS mode these tests should pass,
  // because 0x7ff8000000000000 is still the exact expected NaN.
  status |= test__divdf3(0x0000000000000000, 0x0000000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x0000000000000000, 0x8000000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0x7ff0000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x8000000000000000, 0x0000000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x8000000000000000, 0x8000000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0xfff0000000000000, 0x7ff0000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0xfff0000000000000, 0xfff0000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x3ff0000000000000, 0x7ff8000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x7ff8000000000000, 0x3ff0000000000000, 0x7ff8000000000000);
  status |= test__divdf3(0x7ff8000000000000, 0x7ff8000000000000, 0x7ff8000000000000);

#ifdef ARM_NAN_HANDLING
  // Tests specific to the NaN handling of Arm hardware, mimicked by
  // arm/divdf3.S:
  //
  //  - a quiet NaN is distinguished by the top mantissa bit being 1
  //
  //  - if a signalling NaN appears in the input, the output quiet NaN is
  //    obtained by setting its top mantissa bit and leaving everything else
  //    unchanged
  //
  //  - if both operands are signalling NaNs then the output NaN is derived
  //    from the first operand
  //
  //  - if both operands are quiet NaNs then the output NaN is the first
  //    operand
  //
  //  - invalid operations not involving an input NaN return the quiet
  //    NaN with fewest bits set, 0x7ff8000000000000.
  status |= test__divdf3(0x0000000000000000, 0x7ff3758244400801, 0x7ffb758244400801);
  status |= test__divdf3(0x0000000000000000, 0x7fff44d3f65148af, 0x7fff44d3f65148af);
  status |= test__divdf3(0x0000000000000001, 0x7ff48607b4b37057, 0x7ffc8607b4b37057);
  status |= test__divdf3(0x0000000000000001, 0x7ff855f2d435b33d, 0x7ff855f2d435b33d);
  status |= test__divdf3(0x000fffffffffffff, 0x7ff169269a674e13, 0x7ff969269a674e13);
  status |= test__divdf3(0x000fffffffffffff, 0x7ffc80978b2ef0da, 0x7ffc80978b2ef0da);
  status |= test__divdf3(0x3ff0000000000000, 0x7ff3458ad034593d, 0x7ffb458ad034593d);
  status |= test__divdf3(0x3ff0000000000000, 0x7ffdd8bb98c9f13a, 0x7ffdd8bb98c9f13a);
  status |= test__divdf3(0x7fefffffffffffff, 0x7ff79a8b96250a98, 0x7fff9a8b96250a98);
  status |= test__divdf3(0x7fefffffffffffff, 0x7ffdcc675b63bb94, 0x7ffdcc675b63bb94);
  status |= test__divdf3(0x7ff0000000000000, 0x7ff018cfaf4d0fff, 0x7ff818cfaf4d0fff);
  status |= test__divdf3(0x7ff0000000000000, 0x7ff83ad1ab4dfd24, 0x7ff83ad1ab4dfd24);
  status |= test__divdf3(0x7ff48ce6c0cdd5ac, 0x0000000000000000, 0x7ffc8ce6c0cdd5ac);
  status |= test__divdf3(0x7ff08a34f3d5385b, 0x0000000000000001, 0x7ff88a34f3d5385b);
  status |= test__divdf3(0x7ff0a264c1c96281, 0x000fffffffffffff, 0x7ff8a264c1c96281);
  status |= test__divdf3(0x7ff77ce629e61f0e, 0x3ff0000000000000, 0x7fff7ce629e61f0e);
  status |= test__divdf3(0x7ff715e2d147fd76, 0x7fefffffffffffff, 0x7fff15e2d147fd76);
  status |= test__divdf3(0x7ff689a2031f1781, 0x7ff0000000000000, 0x7ffe89a2031f1781);
  status |= test__divdf3(0x7ff5dfb4a0c8cd05, 0x7ff11c1fe9793a33, 0x7ffddfb4a0c8cd05);
  status |= test__divdf3(0x7ff5826283ffb5d7, 0x7fff609b83884e81, 0x7ffd826283ffb5d7);
  status |= test__divdf3(0x7ff7cb03f2e61d42, 0x8000000000000000, 0x7fffcb03f2e61d42);
  status |= test__divdf3(0x7ff2adc8dfe72c96, 0x8000000000000001, 0x7ffaadc8dfe72c96);
  status |= test__divdf3(0x7ff4fc0bacc707f2, 0x800fffffffffffff, 0x7ffcfc0bacc707f2);
  status |= test__divdf3(0x7ff76248c8c9a619, 0xbff0000000000000, 0x7fff6248c8c9a619);
  status |= test__divdf3(0x7ff367972fce131b, 0xffefffffffffffff, 0x7ffb67972fce131b);
  status |= test__divdf3(0x7ff188f5ac284e92, 0xfff0000000000000, 0x7ff988f5ac284e92);
  status |= test__divdf3(0x7ffed4c22e4e569d, 0x0000000000000000, 0x7ffed4c22e4e569d);
  status |= test__divdf3(0x7ffe95105fa3f339, 0x0000000000000001, 0x7ffe95105fa3f339);
  status |= test__divdf3(0x7ffb8d33dbb9ecfb, 0x000fffffffffffff, 0x7ffb8d33dbb9ecfb);
  status |= test__divdf3(0x7ff874e41dc63e07, 0x3ff0000000000000, 0x7ff874e41dc63e07);
  status |= test__divdf3(0x7ffe27594515ecdf, 0x7fefffffffffffff, 0x7ffe27594515ecdf);
  status |= test__divdf3(0x7ffeac86d5c69bdf, 0x7ff0000000000000, 0x7ffeac86d5c69bdf);
  status |= test__divdf3(0x7ff97d657b99f76f, 0x7ff7e4149862a796, 0x7fffe4149862a796);
  status |= test__divdf3(0x7ffad17c6aa33fad, 0x7ffd898893ad4d28, 0x7ffad17c6aa33fad);
  status |= test__divdf3(0x7ff96e04e9c3d173, 0x8000000000000000, 0x7ff96e04e9c3d173);
  status |= test__divdf3(0x7ffec01ad8da3abb, 0x8000000000000001, 0x7ffec01ad8da3abb);
  status |= test__divdf3(0x7ffd1d565c495941, 0x800fffffffffffff, 0x7ffd1d565c495941);
  status |= test__divdf3(0x7ffe3d24f1e474a7, 0xbff0000000000000, 0x7ffe3d24f1e474a7);
  status |= test__divdf3(0x7ffc206f2bb8c8ce, 0xffefffffffffffff, 0x7ffc206f2bb8c8ce);
  status |= test__divdf3(0x7ff93efdecfb7d3b, 0xfff0000000000000, 0x7ff93efdecfb7d3b);
  status |= test__divdf3(0x8000000000000000, 0x7ff2ee725d143ac5, 0x7ffaee725d143ac5);
  status |= test__divdf3(0x8000000000000000, 0x7ffbba26e5c5fe98, 0x7ffbba26e5c5fe98);
  status |= test__divdf3(0x8000000000000001, 0x7ff7818a1cd26df9, 0x7fff818a1cd26df9);
  status |= test__divdf3(0x8000000000000001, 0x7ffaee6cc63b5292, 0x7ffaee6cc63b5292);
  status |= test__divdf3(0x800fffffffffffff, 0x7ff401096edaf79d, 0x7ffc01096edaf79d);
  status |= test__divdf3(0x800fffffffffffff, 0x7ffbf1778c7a2e59, 0x7ffbf1778c7a2e59);
  status |= test__divdf3(0xbff0000000000000, 0x7ff2e8fb0201c496, 0x7ffae8fb0201c496);
  status |= test__divdf3(0xbff0000000000000, 0x7ffcb6a5adb2e154, 0x7ffcb6a5adb2e154);
  status |= test__divdf3(0xffefffffffffffff, 0x7ff1ea1bfc15d71d, 0x7ff9ea1bfc15d71d);
  status |= test__divdf3(0xffefffffffffffff, 0x7ffae0766e21efc0, 0x7ffae0766e21efc0);
  status |= test__divdf3(0xfff0000000000000, 0x7ff3b364cffbdfe6, 0x7ffbb364cffbdfe6);
  status |= test__divdf3(0xfff0000000000000, 0x7ffd0d3223334ae3, 0x7ffd0d3223334ae3);

#endif // ARM_NAN_HANDLING

  return status;
}
