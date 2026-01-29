// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_muldf3

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

// Returns: a * b
COMPILER_RT_ABI double __muldf3(double a, double b);

int test__muldf3(int line, uint64_t a_rep, uint64_t b_rep, uint64_t expected_rep) {
  double a = fromRep64(a_rep), b = fromRep64(b_rep);
  double x = __muldf3(a, b);
#ifdef EXPECT_EXACT_RESULTS
  int ret = toRep64(x) != expected_rep;
#else
  int ret = compareResultD(x, expected_rep);
#endif

  if (ret) {
    printf("error at line %d: __muldf3(%016" PRIx64 ", %016" PRIx64 ") = %016" PRIx64
           ", expected %016" PRIx64 "\n",
           line, a_rep, b_rep, toRep64(x), expected_rep);
  }
  return ret;
}

#define test__muldf3(a,b,x) test__muldf3(__LINE__,a,b,x)

int main(void) {
  int status = 0;

  status |= test__muldf3(0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x000fffffffffffff, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x001fffffffffffff, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x3ff0000000000000, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x7fdfffffffffffff, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x8000000000000000, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x8000000000000002, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x800fffffffffffff, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x8010000000000001, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0x8020000000000000, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0xc008000000000000, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0xffcfffffffffffff, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0xffe0000000000000, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000000, 0xffefffffffffffff, 0x8000000000000000);
  status |= test__muldf3(0x0000000000000001, 0x0000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000001, 0x0000000000000001, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000001, 0x3fe0000000000000, 0x0000000000000000);
  status |= test__muldf3(0x0000000000000001, 0x3fefffffffffffff, 0x0000000000000001);
  status |= test__muldf3(0x0000000000000001, 0x3ff0000000000000, 0x0000000000000001);
  status |= test__muldf3(0x0000000000000001, 0x4000000000000000, 0x0000000000000002);
  status |= test__muldf3(0x0000000000000001, 0x7ff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x0000000000000001, 0xbfefffffffffffff, 0x8000000000000001);
  status |= test__muldf3(0x0000000000000006, 0x3fe0000000000000, 0x0000000000000003);
  status |= test__muldf3(0x0000000000000006, 0xbfe0000000000000, 0x8000000000000003);
  status |= test__muldf3(0x0000000000000008, 0x3fc0000000000000, 0x0000000000000001);
  status |= test__muldf3(0x000ffffffffffff7, 0x8020000000000003, 0x8000000000000000);
  status |= test__muldf3(0x000ffffffffffff8, 0x3ff0000000000001, 0x000ffffffffffff9);
  status |= test__muldf3(0x000ffffffffffff8, 0x3ff0000000000008, 0x0010000000000000);
  status |= test__muldf3(0x000ffffffffffff8, 0xbff0000000000001, 0x800ffffffffffff9);
  status |= test__muldf3(0x000ffffffffffff8, 0xbff0000000000008, 0x8010000000000000);
  status |= test__muldf3(0x000ffffffffffffc, 0x4000000000000000, 0x001ffffffffffff8);
  status |= test__muldf3(0x000ffffffffffffe, 0x3feffffffffffffc, 0x000ffffffffffffc);
  status |= test__muldf3(0x000ffffffffffffe, 0x3ff0000000000001, 0x000fffffffffffff);
  status |= test__muldf3(0x000ffffffffffffe, 0xbff0000000000001, 0x800fffffffffffff);
  status |= test__muldf3(0x000fffffffffffff, 0x000ffffffffffffe, 0x0000000000000000);
  status |= test__muldf3(0x000fffffffffffff, 0x3cb0000000000001, 0x0000000000000001);
  status |= test__muldf3(0x000fffffffffffff, 0x3fe0000000000001, 0x0008000000000000);
  status |= test__muldf3(0x000fffffffffffff, 0x3ff0000000000001, 0x0010000000000000);
  status |= test__muldf3(0x000fffffffffffff, 0x4000000000000000, 0x001ffffffffffffe);
  status |= test__muldf3(0x0010000000000000, 0x0000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x0010000000000000, 0x0010000000000000, 0x0000000000000000);
  status |= test__muldf3(0x0010000000000000, 0x3feffffffffffffe, 0x000fffffffffffff);
  status |= test__muldf3(0x0010000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x0010000000000000, 0x8010000000000000, 0x8000000000000000);
  status |= test__muldf3(0x0010000000000000, 0xc000000000000000, 0x8020000000000000);
  status |= test__muldf3(0x0010000000000001, 0x3feffffffffffffa, 0x000ffffffffffffe);
  status |= test__muldf3(0x0010000000000001, 0x3feffffffffffffe, 0x0010000000000000);
  status |= test__muldf3(0x0010000000000001, 0xc000000000000000, 0x8020000000000001);
  status |= test__muldf3(0x0010000000000002, 0x3feffffffffffffc, 0x0010000000000000);
  status |= test__muldf3(0x001ffffffffffff8, 0x3fe0000000000000, 0x000ffffffffffffc);
  status |= test__muldf3(0x001ffffffffffffe, 0x3fe0000000000000, 0x000fffffffffffff);
  status |= test__muldf3(0x001ffffffffffffe, 0xbfe0000000000000, 0x800fffffffffffff);
  status |= test__muldf3(0x001fffffffffffff, 0x3fe0000000000000, 0x0010000000000000);
  status |= test__muldf3(0x001fffffffffffff, 0xbfe0000000000000, 0x8010000000000000);
  status |= test__muldf3(0x3fe0000000000000, 0x8000000000000001, 0x8000000000000000);
  status |= test__muldf3(0x3ff0000000000000, 0x000ffffffffffffd, 0x000ffffffffffffd);
  status |= test__muldf3(0x3ff0000000000000, 0x0020000000000003, 0x0020000000000003);
  status |= test__muldf3(0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000);
  status |= test__muldf3(0x3ff0000000000000, 0x4000000000000000, 0x4000000000000000);
  status |= test__muldf3(0x3ff0000000000000, 0x8000000000000001, 0x8000000000000001);
  status |= test__muldf3(0x3ff0000000000000, 0x8000000000000009, 0x8000000000000009);
  status |= test__muldf3(0x3ff0000000000001, 0x3ff0000000000001, 0x3ff0000000000002);
  status |= test__muldf3(0x3ff0000000000001, 0xbff0000000000001, 0xbff0000000000002);
  status |= test__muldf3(0x3ff0000000000001, 0xbff0000000000002, 0xbff0000000000003);
  status |= test__muldf3(0x3ff0000000000002, 0x3ff0000000000001, 0x3ff0000000000003);
  status |= test__muldf3(0x3ff0000000000002, 0x7feffffffffffffe, 0x7ff0000000000000);
  status |= test__muldf3(0x3ff0000000000001, 0x7feffffffffffffe, 0x7ff0000000000000);
  status |= test__muldf3(0x4000000000000000, 0x0010000000000000, 0x0020000000000000);
  status |= test__muldf3(0x4000000000000000, 0x0010000000000001, 0x0020000000000001);
  status |= test__muldf3(0x4000000000000000, 0x3ff0000000000000, 0x4000000000000000);
  status |= test__muldf3(0x4000000000000000, 0x4008000000000000, 0x4018000000000000);
  status |= test__muldf3(0x4000000000000000, 0x7fd0000000000000, 0x7fe0000000000000);
  status |= test__muldf3(0x4000000000000000, 0x7fdfffffffffffff, 0x7fefffffffffffff);
  status |= test__muldf3(0x4000000000000000, 0x800ffffffffffffd, 0x801ffffffffffffa);
  status |= test__muldf3(0x4000000000000000, 0x8010000000000003, 0x8020000000000003);
  status |= test__muldf3(0x4000000000000000, 0x8010000000000005, 0x8020000000000005);
  status |= test__muldf3(0x4000000000000000, 0xbff0000000000000, 0xc000000000000000);
  status |= test__muldf3(0x4000000000000000, 0xffcffffffffffffd, 0xffdffffffffffffd);
  status |= test__muldf3(0x4000000000000000, 0xffd0000000000003, 0xffe0000000000003);
  status |= test__muldf3(0x4007ffffffffffff, 0x3feffffffffffffd, 0x4007fffffffffffd);
  status |= test__muldf3(0x4007ffffffffffff, 0x3feffffffffffffe, 0x4007fffffffffffe);
  status |= test__muldf3(0x4007ffffffffffff, 0x3fefffffffffffff, 0x4007fffffffffffe);
  status |= test__muldf3(0x4007ffffffffffff, 0xbfeffffffffffffd, 0xc007fffffffffffd);
  status |= test__muldf3(0x4008000000000000, 0x0000000000000002, 0x0000000000000006);
  status |= test__muldf3(0x4008000000000000, 0x4000000000000000, 0x4018000000000000);
  status |= test__muldf3(0x4008000000000000, 0x4008000000000000, 0x4022000000000000);
  status |= test__muldf3(0x4008000000000000, 0xc000000000000000, 0xc018000000000000);
  status |= test__muldf3(0x4008000000000001, 0x3ff0000000000001, 0x4008000000000003);
  status |= test__muldf3(0x4008000000000001, 0x3ff0000000000003, 0x4008000000000006);
  status |= test__muldf3(0x4008000000000001, 0xbff0000000000003, 0xc008000000000006);
  status |= test__muldf3(0x4010000000000000, 0x0000000000000002, 0x0000000000000008);
  status |= test__muldf3(0x4010000000000000, 0x7fcfffffffffffff, 0x7fefffffffffffff);
  status |= test__muldf3(0x4010000000000000, 0xffcfffffffffffff, 0xffefffffffffffff);
  status |= test__muldf3(0x4013ffffffffffff, 0x3fefffffffffffff, 0x4013fffffffffffe);
  status |= test__muldf3(0x4014000000000000, 0x0000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x4014000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x4014000000000001, 0x3ff0000000000001, 0x4014000000000002);
  status |= test__muldf3(0x401bffffffffffff, 0x3feffffffffffffc, 0x401bfffffffffffc);
  status |= test__muldf3(0x401bffffffffffff, 0x3fefffffffffffff, 0x401bfffffffffffe);
  status |= test__muldf3(0x401c000000000000, 0x8000000000000000, 0x8000000000000000);
  status |= test__muldf3(0x401c000000000000, 0xfff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x401c000000000001, 0x3ff0000000000001, 0x401c000000000003);
  status |= test__muldf3(0x7fcffffffffffffd, 0x4010000000000000, 0x7feffffffffffffd);
  status |= test__muldf3(0x7fcffffffffffffd, 0xc010000000000000, 0xffeffffffffffffd);
  status |= test__muldf3(0x7fd0000000000000, 0xc000000000000000, 0xffe0000000000000);
  status |= test__muldf3(0x7fdffffffffffffd, 0xc000000000000008, 0xfff0000000000000);
  status |= test__muldf3(0x7fdfffffffffffff, 0xc000000000000000, 0xffefffffffffffff);
  status |= test__muldf3(0x7fe0000000000000, 0x0000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x7fe0000000000000, 0x4000000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x7fe0000000000000, 0x7fe0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x7fe0000000000000, 0x7feffffffffffffe, 0x7ff0000000000000);
  status |= test__muldf3(0x7fe0000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x7fe0000000000000, 0xffd0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x7fe0000000000000, 0xffd0000000000004, 0xfff0000000000000);
  status |= test__muldf3(0x7fe0000000000000, 0xffe0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x7fe0000000000009, 0x7feffffffffffffa, 0x7ff0000000000000);
  status |= test__muldf3(0x7fe0000000000009, 0xc018000000000002, 0xfff0000000000000);
  status |= test__muldf3(0x7fefffffffffffff, 0x0000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x000fffffffffffff, 0x7ff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x001fffffffffffff, 0x7ff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x3ff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x7fdfffffffffffff, 0x7ff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x8000000000000002, 0xfff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x800fffffffffffff, 0xfff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x8010000000000001, 0xfff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x8020000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0xc008000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0xffe0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0xffefffffffffffff, 0xfff0000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0xfff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x8000000000000000, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0x8000000000000000, 0x4018000000000000, 0x8000000000000000);
  status |= test__muldf3(0x8000000000000000, 0x7fefffffffffffff, 0x8000000000000000);
  status |= test__muldf3(0x8000000000000000, 0x8000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x8000000000000000, 0x8000000000000004, 0x0000000000000000);
  status |= test__muldf3(0x8000000000000000, 0x8010000000000000, 0x0000000000000000);
  status |= test__muldf3(0x8000000000000000, 0xc020000000000000, 0x0000000000000000);
  status |= test__muldf3(0x8000000000000000, 0xffd0000000000000, 0x0000000000000000);
  status |= test__muldf3(0x8000000000000001, 0x0000000000000001, 0x8000000000000000);
  status |= test__muldf3(0x8000000000000001, 0x4014000000000000, 0x8000000000000005);
  status |= test__muldf3(0x8000000000000002, 0x3ff0000000000000, 0x8000000000000002);
  status |= test__muldf3(0x8000000000000003, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0x8000000000000003, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x8000000000000004, 0xbff0000000000000, 0x0000000000000004);
  status |= test__muldf3(0x8000000000000008, 0x3fc0000000000000, 0x8000000000000001);
  status |= test__muldf3(0x800ffffffffffff7, 0x0020000000000003, 0x8000000000000000);
  status |= test__muldf3(0x800ffffffffffff7, 0x3ff0000000000001, 0x800ffffffffffff8);
  status |= test__muldf3(0x800ffffffffffffd, 0xc000000000000000, 0x001ffffffffffffa);
  status |= test__muldf3(0x800fffffffffffff, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0x800fffffffffffff, 0x3ff0000000000001, 0x8010000000000000);
  status |= test__muldf3(0x800fffffffffffff, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x800fffffffffffff, 0x8000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x800fffffffffffff, 0x800ffffffffffffe, 0x0000000000000000);
  status |= test__muldf3(0x800fffffffffffff, 0xbff0000000000000, 0x000fffffffffffff);
  status |= test__muldf3(0x800fffffffffffff, 0xfff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x8010000000000000, 0x0010000000000000, 0x8000000000000000);
  status |= test__muldf3(0x8010000000000000, 0x8010000000000000, 0x0000000000000000);
  status |= test__muldf3(0x8010000000000001, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0x8010000000000001, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0x8010000000000001, 0xbff0000000000000, 0x0010000000000001);
  status |= test__muldf3(0x801ffffffffffffc, 0x3fe0000000000000, 0x800ffffffffffffe);
  status |= test__muldf3(0x801ffffffffffffc, 0xbfe0000000000000, 0x000ffffffffffffe);
  status |= test__muldf3(0x801ffffffffffffe, 0x3ff0000000000000, 0x801ffffffffffffe);
  status |= test__muldf3(0x801fffffffffffff, 0x8000000000000000, 0x0000000000000000);
  status |= test__muldf3(0x801fffffffffffff, 0xfff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x8020000000000000, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0x8020000000000000, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xbfefffffffffffff, 0xffefffffffffffff, 0x7feffffffffffffe);
  status |= test__muldf3(0xbff0000000000000, 0x0000000000000009, 0x8000000000000009);
  status |= test__muldf3(0xbff0000000000000, 0x0010000000000009, 0x8010000000000009);
  status |= test__muldf3(0xbff0000000000000, 0x3ff0000000000000, 0xbff0000000000000);
  status |= test__muldf3(0xbff0000000000000, 0x4000000000000000, 0xc000000000000000);
  status |= test__muldf3(0xbff0000000000000, 0xbff0000000000000, 0x3ff0000000000000);
  status |= test__muldf3(0xbff0000000000000, 0xc000000000000000, 0x4000000000000000);
  status |= test__muldf3(0xbff0000000000001, 0x3ff0000000000001, 0xbff0000000000002);
  status |= test__muldf3(0xbff0000000000001, 0xbff0000000000001, 0x3ff0000000000002);
  status |= test__muldf3(0xbff0000000000001, 0xbff0000000000002, 0x3ff0000000000003);
  status |= test__muldf3(0xbff0000000000002, 0x3ff0000000000001, 0xbff0000000000003);
  status |= test__muldf3(0xbff0000000000002, 0xbff0000000000001, 0x3ff0000000000003);
  status |= test__muldf3(0xc000000000000000, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0xc000000000000000, 0x000ffffffffffffd, 0x801ffffffffffffa);
  status |= test__muldf3(0xc000000000000000, 0x0010000000000001, 0x8020000000000001);
  status |= test__muldf3(0xc000000000000000, 0x0010000000000005, 0x8020000000000005);
  status |= test__muldf3(0xc000000000000000, 0x0010000000000009, 0x8020000000000009);
  status |= test__muldf3(0xc000000000000000, 0x4008000000000000, 0xc018000000000000);
  status |= test__muldf3(0xc000000000000000, 0x7fcfffffffffffff, 0xffdfffffffffffff);
  status |= test__muldf3(0xc000000000000000, 0x7fd0000000000001, 0xffe0000000000001);
  status |= test__muldf3(0xc000000000000000, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xc000000000000000, 0xbff0000000000000, 0x4000000000000000);
  status |= test__muldf3(0xc000000000000000, 0xc008000000000000, 0x4018000000000000);
  status |= test__muldf3(0xc007fffffffffffe, 0x7fe0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xc007ffffffffffff, 0x3fefffffffffffff, 0xc007fffffffffffe);
  status |= test__muldf3(0xc008000000000000, 0x4008000000000000, 0xc022000000000000);
  status |= test__muldf3(0xc008000000000000, 0xc000000000000000, 0x4018000000000000);
  status |= test__muldf3(0xc008000000000000, 0xc008000000000000, 0x4022000000000000);
  status |= test__muldf3(0xc008000000000000, 0xffe0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xc008000000000001, 0x3ff0000000000001, 0xc008000000000003);
  status |= test__muldf3(0xc010000000000000, 0x7fcfffffffffffff, 0xffefffffffffffff);
  status |= test__muldf3(0xc010000000000000, 0x8000000000000000, 0x0000000000000000);
  status |= test__muldf3(0xc010000000000000, 0xffcfffffffffffff, 0x7fefffffffffffff);
  status |= test__muldf3(0xc010000000000000, 0xfff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xc013fffffffffffe, 0xffe0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xc013ffffffffffff, 0xbfefffffffffffff, 0x4013fffffffffffe);
  status |= test__muldf3(0xc014000000000001, 0xbff0000000000001, 0x4014000000000002);
  status |= test__muldf3(0xc01bfffffffffff9, 0x7fe0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xc022000000000000, 0x7fe0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xc022000000000001, 0xffe0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xffcffffffffffff9, 0x7fe0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xffcffffffffffff9, 0xc00fffffffffffff, 0x7feffffffffffff8);
  status |= test__muldf3(0xffcffffffffffffd, 0x4010000000000000, 0xffeffffffffffffd);
  status |= test__muldf3(0xffcffffffffffffd, 0xc010000000000000, 0x7feffffffffffffd);
  status |= test__muldf3(0xffcfffffffffffff, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0xffcfffffffffffff, 0x4000000000000001, 0xffe0000000000000);
  status |= test__muldf3(0xffcfffffffffffff, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xffd0000000000000, 0x0000000000000000, 0x8000000000000000);
  status |= test__muldf3(0xffd0000000000000, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xffdffffffffffff7, 0x7fd0000000000001, 0xfff0000000000000);
  status |= test__muldf3(0xffdfffffffffffff, 0x3ff0000000000001, 0xffe0000000000000);
  status |= test__muldf3(0xffdfffffffffffff, 0x8000000000000000, 0x0000000000000000);
  status |= test__muldf3(0xffe0000000000005, 0xffe0000000000001, 0x7ff0000000000000);
  status |= test__muldf3(0xffeffffffffffffd, 0x7fe0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xffeffffffffffffd, 0xc008000000000001, 0x7ff0000000000000);
  status |= test__muldf3(0xffeffffffffffffd, 0xffe0000000000001, 0x7ff0000000000000);
  status |= test__muldf3(0xffefffffffffffff, 0x8000000000000000, 0x0000000000000000);
  status |= test__muldf3(0xffefffffffffffff, 0xffefffffffffffff, 0x7ff0000000000000);
  status |= test__muldf3(0xffefffffffffffff, 0xfff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xfff0000000000000, 0x4018000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xfff0000000000000, 0x7ff0000000000000, 0xfff0000000000000);
  status |= test__muldf3(0xfff0000000000000, 0x8000000000000004, 0x7ff0000000000000);
  status |= test__muldf3(0xfff0000000000000, 0x8010000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xfff0000000000000, 0xc020000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xfff0000000000000, 0xffd0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0xfff0000000000000, 0xfff0000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x002ffffffe000000, 0x3fcffffffffffffd, 0x000ffffffeffffff);
  status |= test__muldf3(0xbfeffeffffffffff, 0x8010000000000100, 0x000fff80000000ff);
  status |= test__muldf3(0x802ffffffe000000, 0x3fcffffffffffffd, 0x800ffffffeffffff);
  status |= test__muldf3(0xbfeffeffffffffff, 0x0010000000000100, 0x800fff80000000ff);
  status |= test__muldf3(0xbf9e8325a5aa6c8d, 0xbf9e8325a5aa6c8d, 0x3f4d180013083955);
  status |= test__muldf3(0x3ffd25d7ea4fa2d4, 0x3fe4000000000000, 0x3ff237a6f271c5c4);
  status |= test__muldf3(0x6ffd25d7ea4fa2d4, 0x4fe4000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x201d25d7ea4fa2d4, 0x1fd4000000000000, 0x00091bd37938e2e2);
  status |= test__muldf3(0x3ffd25d7ea4fa2d4, 0x3fe8000000000000, 0x3ff5dc61efbbba1f);
  status |= test__muldf3(0x6ffd25d7ea4fa2d4, 0x4fe8000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x201d25d7ea4fa2d4, 0x1fd8000000000000, 0x000aee30f7dddd10);
  status |= test__muldf3(0x3ffd25d7ea4fa2d4, 0x3fec000000000000, 0x3ff9811ced05ae7a);
  status |= test__muldf3(0x6ffd25d7ea4fa2d4, 0x4fec000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x201d25d7ea4fa2d4, 0x1fdc000000000000, 0x000cc08e7682d73d);
  status |= test__muldf3(0x3ff265f139b6c87c, 0x3ff7000000000000, 0x3ffa728ac2f6c032);
  status |= test__muldf3(0x6ff265f139b6c87c, 0x4ff7000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x201265f139b6c87c, 0x1fe7000000000000, 0x000d3945617b6019);
  status |= test__muldf3(0x3ff265f139b6c87c, 0x3ff5000000000000, 0x3ff825cc9bbfe723);
  status |= test__muldf3(0x6ff265f139b6c87c, 0x4ff5000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x201265f139b6c87c, 0x1fe5000000000000, 0x000c12e64ddff391);
  status |= test__muldf3(0x3ffe5ab1dc9f12f9, 0x3ff0c1a10c80f0b7, 0x3fffca09666ab16e);
  status |= test__muldf3(0x6ffe5ab1dc9f12f9, 0x4ff0c1a10c80f0b7, 0x7ff0000000000000);
  status |= test__muldf3(0x201e5ab1dc9f12f9, 0x1fe0c1a10c80f0b7, 0x000fe504b33558b7);
  status |= test__muldf3(0x3ffe5ab1dc9f12f9, 0x3fe73e5ef37f0f49, 0x3ff60c59a0917f00);
  status |= test__muldf3(0x6ffe5ab1dc9f12f9, 0x4fe73e5ef37f0f49, 0x7ff0000000000000);
  status |= test__muldf3(0x201e5ab1dc9f12f9, 0x1fd73e5ef37f0f49, 0x000b062cd048bf80);
  status |= test__muldf3(0x3ffe5ab1dc9f12f9, 0x3fe8c1a10c80f0b7, 0x3ff77bb12a5d1d75);
  status |= test__muldf3(0x6ffe5ab1dc9f12f9, 0x4fe8c1a10c80f0b7, 0x7ff0000000000000);
  status |= test__muldf3(0x201e5ab1dc9f12f9, 0x1fd8c1a10c80f0b7, 0x000bbdd8952e8ebb);
  status |= test__muldf3(0x3ffc6be665de3b1d, 0x3fe52d156619a0cb, 0x3ff2ced9f056fba8);
  status |= test__muldf3(0x6ffc6be665de3b1d, 0x4fe52d156619a0cb, 0x7ff0000000000000);
  status |= test__muldf3(0x201c6be665de3b1d, 0x1fd52d156619a0cb, 0x0009676cf82b7dd4);
  status |= test__muldf3(0x3ffc6be665de3b1d, 0x3fead2ea99e65f35, 0x3ff7d2ffa8765d03);
  status |= test__muldf3(0x6ffc6be665de3b1d, 0x4fead2ea99e65f35, 0x7ff0000000000000);
  status |= test__muldf3(0x201c6be665de3b1d, 0x1fdad2ea99e65f35, 0x000be97fd43b2e82);
  status |= test__muldf3(0x3ff1c0635d3cd39d, 0x3ff5c9b956d0b54b, 0x3ff82c50eb71ac34);
  status |= test__muldf3(0x6ff1c0635d3cd39d, 0x4ff5c9b956d0b54b, 0x7ff0000000000000);
  status |= test__muldf3(0x2011c0635d3cd39d, 0x1fe5c9b956d0b54b, 0x000c162875b8d61a);
  status |= test__muldf3(0x3ff1c0635d3cd39d, 0x3ff23646a92f4ab5, 0x3ff434a77da664d4);
  status |= test__muldf3(0x6ff1c0635d3cd39d, 0x4ff23646a92f4ab5, 0x7ff0000000000000);
  status |= test__muldf3(0x2011c0635d3cd39d, 0x1fe23646a92f4ab5, 0x000a1a53bed3326a);
  status |= test__muldf3(0x3ff1c0635d3cd39d, 0x3ffa3646a92f4ab5, 0x3ffd14d92c44cea3);
  status |= test__muldf3(0x6ff1c0635d3cd39d, 0x4ffa3646a92f4ab5, 0x7ff0000000000000);
  status |= test__muldf3(0x2011c0635d3cd39d, 0x1fea3646a92f4ab5, 0x000e8a6c96226751);
  status |= test__muldf3(0x3ff1c0635d3cd39d, 0x3ff1c9b956d0b54b, 0x3ff3bc381422774d);
  status |= test__muldf3(0x6ff1c0635d3cd39d, 0x4ff1c9b956d0b54b, 0x7ff0000000000000);
  status |= test__muldf3(0x2011c0635d3cd39d, 0x1fe1c9b956d0b54b, 0x0009de1c0a113ba6);
  status |= test__muldf3(0x3ff907065fd11389, 0x3fe46bad37af52b9, 0x3feff135e5756ec7);
  status |= test__muldf3(0x6ff907065fd11389, 0x4fe46bad37af52b9, 0x7feff135e5756ec7);
  status |= test__muldf3(0x201907065fd11389, 0x1fd46bad37af52b9, 0x0007fc4d795d5bb2);
  status |= test__muldf3(0x3ff907065fd11389, 0x3feb9452c850ad47, 0x3ff591ee9cfee5ea);
  status |= test__muldf3(0x6ff907065fd11389, 0x4feb9452c850ad47, 0x7ff0000000000000);
  status |= test__muldf3(0x201907065fd11389, 0x1fdb9452c850ad47, 0x000ac8f74e7f72f5);
  status |= test__muldf3(0x3ff761c03e198df7, 0x3fe7f47c731d43c7, 0x3ff180e675617e83);
  status |= test__muldf3(0x6ff761c03e198df7, 0x4fe7f47c731d43c7, 0x7ff0000000000000);
  status |= test__muldf3(0x201761c03e198df7, 0x1fd7f47c731d43c7, 0x0008c0733ab0bf41);
  status |= test__muldf3(0x3ffce6d1246c46fb, 0x3ff0b3469ded2bcd, 0x3ffe2aa6f74c0ffd);
  status |= test__muldf3(0x6ffce6d1246c46fb, 0x4ff0b3469ded2bcd, 0x7ff0000000000000);
  status |= test__muldf3(0x201ce6d1246c46fb, 0x1fe0b3469ded2bcd, 0x000f15537ba607fe);
  status |= test__muldf3(0x3ffd5701100ec79d, 0x3fee654fee13094b, 0x3ffbde74e37bb583);
  status |= test__muldf3(0x6ffd5701100ec79d, 0x4fee654fee13094b, 0x7ff0000000000000);
  status |= test__muldf3(0x201d5701100ec79d, 0x1fde654fee13094b, 0x000def3a71bddac1);
  status |= test__muldf3(0x3ffce1a06e8bcfd3, 0x3ff01c54436a605b, 0x3ffd14c361885d61);
  status |= test__muldf3(0x6ffce1a06e8bcfd3, 0x4ff01c54436a605b, 0x7ff0000000000000);
  status |= test__muldf3(0x201ce1a06e8bcfd3, 0x1fe01c54436a605b, 0x000e8a61b0c42eb0);
  status |= test__muldf3(0x3ff21d1a5ca518a5, 0x3ff29f0ce1150f2d, 0x3ff514cd72d743f2);
  status |= test__muldf3(0x6ff21d1a5ca518a5, 0x4ff29f0ce1150f2d, 0x7ff0000000000000);
  status |= test__muldf3(0x20121d1a5ca518a5, 0x1fe29f0ce1150f2d, 0x000a8a66b96ba1f9);
  status |= test__muldf3(0x3ff031a98dbf97ba, 0x3ff4000000000000, 0x3ff43e13f12f7da8);
  status |= test__muldf3(0x6ff031a98dbf97ba, 0x4ff4000000000000, 0x7ff0000000000000);
  status |= test__muldf3(0x201031a98dbf97ba, 0x1fe4000000000000, 0x000a1f09f897bed4);
  status |= test__muldf3(0x0000000000000003, 0xc00fffffffffffff, 0x800000000000000c);
  status |= test__muldf3(0x0000000000000003, 0x400fffffffffffff, 0x000000000000000c);
  status |= test__muldf3(0x8000000000000003, 0xc00fffffffffffff, 0x000000000000000c);
  status |= test__muldf3(0x8000000000000003, 0x400fffffffffffff, 0x800000000000000c);
  status |= test__muldf3(0x0000000000000003, 0xc00ffffffffffffd, 0x800000000000000c);
  status |= test__muldf3(0x0000000000000003, 0x400ffffffffffffd, 0x000000000000000c);
  status |= test__muldf3(0x8000000000000003, 0xc00ffffffffffffd, 0x000000000000000c);
  status |= test__muldf3(0x8000000000000003, 0x400ffffffffffffd, 0x800000000000000c);
  status |= test__muldf3(0x1e51f703ee090000, 0x1e5c8000e4000000, 0x0000000000000001);
  status |= test__muldf3(0x1e561ed9745fdb21, 0x1e57255ca25b68e1, 0x0000000000000001);
  status |= test__muldf3(0x7feffffffff00000, 0xc000000000080000, 0xfff0000000000000);

  // Test that the result of an operation is a NaN at all when it should be.
  //
  // In most configurations these tests' results are checked compared using
  // compareResultD, so we set all the answers to the canonical NaN
  // 0x7ff8000000000000, which causes compareResultF to accept any NaN
  // encoding. We also use the same value as the input NaN in tests that have
  // one, so that even in EXPECT_EXACT_RESULTS mode these tests should pass,
  // because 0x7ff8000000000000 is still the exact expected NaN.
  status |= test__muldf3(0x7ff0000000000000, 0x0000000000000000, 0x7ff8000000000000);
  status |= test__muldf3(0x7ff0000000000000, 0x8000000000000000, 0x7ff8000000000000);
  status |= test__muldf3(0x8000000000000000, 0x7ff0000000000000, 0x7ff8000000000000);
  status |= test__muldf3(0x8000000000000000, 0xfff0000000000000, 0x7ff8000000000000);
  status |= test__muldf3(0x3ff0000000000000, 0x7ff8000000000000, 0x7ff8000000000000);
  status |= test__muldf3(0x7ff8000000000000, 0x3ff0000000000000, 0x7ff8000000000000);
  status |= test__muldf3(0x7ff8000000000000, 0x7ff8000000000000, 0x7ff8000000000000);

#ifdef ARM_NAN_HANDLING
  // Tests specific to the NaN handling of Arm hardware, mimicked by
  // arm/muldf3.S:
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
  status |= test__muldf3(0x0000000000000000, 0x7ff3758244400801, 0x7ffb758244400801);
  status |= test__muldf3(0x0000000000000000, 0x7fff44d3f65148af, 0x7fff44d3f65148af);
  status |= test__muldf3(0x0000000000000001, 0x7ff48607b4b37057, 0x7ffc8607b4b37057);
  status |= test__muldf3(0x0000000000000001, 0x7ff855f2d435b33d, 0x7ff855f2d435b33d);
  status |= test__muldf3(0x000fffffffffffff, 0x7ff169269a674e13, 0x7ff969269a674e13);
  status |= test__muldf3(0x000fffffffffffff, 0x7ffc80978b2ef0da, 0x7ffc80978b2ef0da);
  status |= test__muldf3(0x3ff0000000000000, 0x7ff3458ad034593d, 0x7ffb458ad034593d);
  status |= test__muldf3(0x3ff0000000000000, 0x7ffdd8bb98c9f13a, 0x7ffdd8bb98c9f13a);
  status |= test__muldf3(0x7fefffffffffffff, 0x7ff79a8b96250a98, 0x7fff9a8b96250a98);
  status |= test__muldf3(0x7fefffffffffffff, 0x7ffdcc675b63bb94, 0x7ffdcc675b63bb94);
  status |= test__muldf3(0x7ff0000000000000, 0x7ff018cfaf4d0fff, 0x7ff818cfaf4d0fff);
  status |= test__muldf3(0x7ff0000000000000, 0x7ff83ad1ab4dfd24, 0x7ff83ad1ab4dfd24);
  status |= test__muldf3(0x7ff48ce6c0cdd5ac, 0x0000000000000000, 0x7ffc8ce6c0cdd5ac);
  status |= test__muldf3(0x7ff08a34f3d5385b, 0x0000000000000001, 0x7ff88a34f3d5385b);
  status |= test__muldf3(0x7ff0a264c1c96281, 0x000fffffffffffff, 0x7ff8a264c1c96281);
  status |= test__muldf3(0x7ff77ce629e61f0e, 0x3ff0000000000000, 0x7fff7ce629e61f0e);
  status |= test__muldf3(0x7ff715e2d147fd76, 0x7fefffffffffffff, 0x7fff15e2d147fd76);
  status |= test__muldf3(0x7ff689a2031f1781, 0x7ff0000000000000, 0x7ffe89a2031f1781);
  status |= test__muldf3(0x7ff5dfb4a0c8cd05, 0x7ff11c1fe9793a33, 0x7ffddfb4a0c8cd05);
  status |= test__muldf3(0x7ff5826283ffb5d7, 0x7fff609b83884e81, 0x7ffd826283ffb5d7);
  status |= test__muldf3(0x7ff7cb03f2e61d42, 0x8000000000000000, 0x7fffcb03f2e61d42);
  status |= test__muldf3(0x7ff2adc8dfe72c96, 0x8000000000000001, 0x7ffaadc8dfe72c96);
  status |= test__muldf3(0x7ff4fc0bacc707f2, 0x800fffffffffffff, 0x7ffcfc0bacc707f2);
  status |= test__muldf3(0x7ff76248c8c9a619, 0xbff0000000000000, 0x7fff6248c8c9a619);
  status |= test__muldf3(0x7ff367972fce131b, 0xffefffffffffffff, 0x7ffb67972fce131b);
  status |= test__muldf3(0x7ff188f5ac284e92, 0xfff0000000000000, 0x7ff988f5ac284e92);
  status |= test__muldf3(0x7ffed4c22e4e569d, 0x0000000000000000, 0x7ffed4c22e4e569d);
  status |= test__muldf3(0x7ffe95105fa3f339, 0x0000000000000001, 0x7ffe95105fa3f339);
  status |= test__muldf3(0x7ffb8d33dbb9ecfb, 0x000fffffffffffff, 0x7ffb8d33dbb9ecfb);
  status |= test__muldf3(0x7ff874e41dc63e07, 0x3ff0000000000000, 0x7ff874e41dc63e07);
  status |= test__muldf3(0x7ffe27594515ecdf, 0x7fefffffffffffff, 0x7ffe27594515ecdf);
  status |= test__muldf3(0x7ffeac86d5c69bdf, 0x7ff0000000000000, 0x7ffeac86d5c69bdf);
  status |= test__muldf3(0x7ff97d657b99f76f, 0x7ff7e4149862a796, 0x7fffe4149862a796);
  status |= test__muldf3(0x7ffad17c6aa33fad, 0x7ffd898893ad4d28, 0x7ffad17c6aa33fad);
  status |= test__muldf3(0x7ff96e04e9c3d173, 0x8000000000000000, 0x7ff96e04e9c3d173);
  status |= test__muldf3(0x7ffec01ad8da3abb, 0x8000000000000001, 0x7ffec01ad8da3abb);
  status |= test__muldf3(0x7ffd1d565c495941, 0x800fffffffffffff, 0x7ffd1d565c495941);
  status |= test__muldf3(0x7ffe3d24f1e474a7, 0xbff0000000000000, 0x7ffe3d24f1e474a7);
  status |= test__muldf3(0x7ffc206f2bb8c8ce, 0xffefffffffffffff, 0x7ffc206f2bb8c8ce);
  status |= test__muldf3(0x7ff93efdecfb7d3b, 0xfff0000000000000, 0x7ff93efdecfb7d3b);
  status |= test__muldf3(0x8000000000000000, 0x7ff2ee725d143ac5, 0x7ffaee725d143ac5);
  status |= test__muldf3(0x8000000000000000, 0x7ffbba26e5c5fe98, 0x7ffbba26e5c5fe98);
  status |= test__muldf3(0x8000000000000001, 0x7ff7818a1cd26df9, 0x7fff818a1cd26df9);
  status |= test__muldf3(0x8000000000000001, 0x7ffaee6cc63b5292, 0x7ffaee6cc63b5292);
  status |= test__muldf3(0x800fffffffffffff, 0x7ff401096edaf79d, 0x7ffc01096edaf79d);
  status |= test__muldf3(0x800fffffffffffff, 0x7ffbf1778c7a2e59, 0x7ffbf1778c7a2e59);
  status |= test__muldf3(0xbff0000000000000, 0x7ff2e8fb0201c496, 0x7ffae8fb0201c496);
  status |= test__muldf3(0xbff0000000000000, 0x7ffcb6a5adb2e154, 0x7ffcb6a5adb2e154);
  status |= test__muldf3(0xffefffffffffffff, 0x7ff1ea1bfc15d71d, 0x7ff9ea1bfc15d71d);
  status |= test__muldf3(0xffefffffffffffff, 0x7ffae0766e21efc0, 0x7ffae0766e21efc0);
  status |= test__muldf3(0xfff0000000000000, 0x7ff3b364cffbdfe6, 0x7ffbb364cffbdfe6);
  status |= test__muldf3(0xfff0000000000000, 0x7ffd0d3223334ae3, 0x7ffd0d3223334ae3);

#endif // ARM_NAN_HANDLING

  return status;
}
