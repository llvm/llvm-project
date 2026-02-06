// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_adddf3

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

// Returns: a + b
COMPILER_RT_ABI double __adddf3(double a, double b);

int test__adddf3(int line, uint64_t a_rep, uint64_t b_rep,
                 uint64_t expected_rep) {
  double a = fromRep64(a_rep), b = fromRep64(b_rep);
  double x = __adddf3(a, b);
#ifdef EXPECT_EXACT_RESULTS
  int ret = toRep64(x) != expected_rep;
#else
  int ret = compareResultD(x, expected_rep);
#endif

  if (ret) {
    printf("error at line %d: __adddf3(%016" PRIx64 ", %016" PRIx64
           ") = %016" PRIx64 ", expected %016" PRIx64 "\n",
           line, a_rep, b_rep, toRep64(x), expected_rep);
  }
  return ret;
}

#define test__adddf3(a, b, x) (test__adddf3)(__LINE__, a, b, x)

int main(void) {
  int status = 0;

  status |=
      test__adddf3(0x0000000000000000, 0x0000000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x0000000000000000, 0x000fffffffffffff, 0x000fffffffffffff);
  status |=
      test__adddf3(0x0000000000000000, 0x3ff0000000000000, 0x3ff0000000000000);
  status |=
      test__adddf3(0x0000000000000000, 0x7fe0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x0000000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x0000000000000000, 0x8000000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x0000000000000000, 0x800fffffffffffff, 0x800fffffffffffff);
  status |=
      test__adddf3(0x0000000000000000, 0x8010000000000000, 0x8010000000000000);
  status |=
      test__adddf3(0x0000000000000000, 0xfff0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0x0000000000000001, 0x0000000000000001, 0x0000000000000002);
  status |=
      test__adddf3(0x0000000000000001, 0x3fefffffffffffff, 0x3fefffffffffffff);
  status |=
      test__adddf3(0x0000000000000001, 0x3ff0000000000000, 0x3ff0000000000000);
  status |=
      test__adddf3(0x0000000000000001, 0x3ffffffffffffffe, 0x3ffffffffffffffe);
  status |=
      test__adddf3(0x0000000000000001, 0x3fffffffffffffff, 0x3fffffffffffffff);
  status |=
      test__adddf3(0x0000000000000001, 0x7fdfffffffffffff, 0x7fdfffffffffffff);
  status |=
      test__adddf3(0x0000000000000001, 0x7fe0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x0000000000000001, 0x7feffffffffffffe, 0x7feffffffffffffe);
  status |=
      test__adddf3(0x0000000000000001, 0x7fefffffffffffff, 0x7fefffffffffffff);
  status |=
      test__adddf3(0x0000000000000001, 0x8000000000000001, 0x0000000000000000);
  status |=
      test__adddf3(0x0000000000000002, 0x8000000000000001, 0x0000000000000001);
  status |=
      test__adddf3(0x0000000000000003, 0x0000000000000000, 0x0000000000000003);
  status |=
      test__adddf3(0x0000000000000003, 0x7ff0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x0000000000000003, 0x8000000000000000, 0x0000000000000003);
  status |=
      test__adddf3(0x0000000000000003, 0x8000000000000002, 0x0000000000000001);
  status |=
      test__adddf3(0x0000000000000003, 0xc014000000000000, 0xc014000000000000);
  status |=
      test__adddf3(0x0000000000000003, 0xffe0000000000000, 0xffe0000000000000);
  status |=
      test__adddf3(0x0000000000000003, 0xfff0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0x0000000000000004, 0x0000000000000004, 0x0000000000000008);
  status |=
      test__adddf3(0x000ffffffffffffc, 0x800ffffffffffffc, 0x0000000000000000);
  status |=
      test__adddf3(0x000ffffffffffffd, 0x800ffffffffffffe, 0x8000000000000001);
  status |=
      test__adddf3(0x000fffffffffffff, 0x000fffffffffffff, 0x001ffffffffffffe);
  status |=
      test__adddf3(0x000fffffffffffff, 0x800ffffffffffffe, 0x0000000000000001);
  status |=
      test__adddf3(0x000fffffffffffff, 0x8010000000000000, 0x8000000000000001);
  status |=
      test__adddf3(0x0010000000000000, 0x0000000000000000, 0x0010000000000000);
  status |=
      test__adddf3(0x0010000000000000, 0x0010000000000000, 0x0020000000000000);
  status |=
      test__adddf3(0x0010000000000000, 0x8010000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x0010000000000001, 0x8010000000000000, 0x0000000000000001);
  status |=
      test__adddf3(0x0010000000000001, 0x8010000000000002, 0x8000000000000001);
  status |=
      test__adddf3(0x001fffffffffffff, 0x8020000000000000, 0x8000000000000001);
  status |=
      test__adddf3(0x001fffffffffffff, 0x8020000000000002, 0x8000000000000005);
  status |=
      test__adddf3(0x001fffffffffffff, 0x8020000000000004, 0x8000000000000009);
  status |=
      test__adddf3(0x0020000000000000, 0x801fffffffffffff, 0x0000000000000001);
  status |=
      test__adddf3(0x0020000000000001, 0x8010000000000001, 0x0010000000000001);
  status |=
      test__adddf3(0x0020000000000001, 0x801fffffffffffff, 0x0000000000000003);
  status |=
      test__adddf3(0x0020000000000002, 0x8010000000000001, 0x0010000000000003);
  status |=
      test__adddf3(0x002fffffffffffff, 0x8030000000000000, 0x8000000000000002);
  status |=
      test__adddf3(0x0030000000000000, 0x802fffffffffffff, 0x0000000000000002);
  status |=
      test__adddf3(0x0030000000000001, 0x802fffffffffffff, 0x0000000000000006);
  status |=
      test__adddf3(0x0030000000000002, 0x8020000000000003, 0x0020000000000001);
  status |=
      test__adddf3(0x3fefffffffffffff, 0x8000000000000001, 0x3fefffffffffffff);
  status |=
      test__adddf3(0x3ff0000000000000, 0x3ff0000000000000, 0x4000000000000000);
  status |=
      test__adddf3(0x3ff0000000000000, 0x3ff0000000000003, 0x4000000000000002);
  status |=
      test__adddf3(0x3ff0000000000000, 0x4000000000000000, 0x4008000000000000);
  status |=
      test__adddf3(0x3ff0000000000000, 0x401c000000000000, 0x4020000000000000);
  status |=
      test__adddf3(0x3ff0000000000000, 0x8000000000000000, 0x3ff0000000000000);
  status |=
      test__adddf3(0x3ff0000000000000, 0xbff0000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x3ff0000000000001, 0x3ff0000000000000, 0x4000000000000000);
  status |=
      test__adddf3(0x3ff0000000000001, 0xbff0000000000000, 0x3cb0000000000000);
  status |=
      test__adddf3(0x3ff0000000000001, 0xbff0000000000002, 0xbcb0000000000000);
  status |=
      test__adddf3(0x3ffffffffffffffc, 0xbffffffffffffffd, 0xbcb0000000000000);
  status |=
      test__adddf3(0x3fffffffffffffff, 0xc000000000000000, 0xbcb0000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0x3cb0000000000000, 0x4000000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0x3ff0000000000000, 0x4008000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0x4000000000000000, 0x4010000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0x4000000000000001, 0x4010000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0xbfffffffffffffff, 0x3cb0000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0xc000000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0xc000000000000001, 0xbcc0000000000000);
  status |=
      test__adddf3(0x4000000000000000, 0xc014000000000000, 0xc008000000000000);
  status |=
      test__adddf3(0x4000000000000001, 0x3cb0000000000000, 0x4000000000000002);
  status |=
      test__adddf3(0x4000000000000001, 0x4000000000000002, 0x4010000000000002);
  status |=
      test__adddf3(0x4000000000000001, 0xbff0000000000001, 0x3ff0000000000001);
  status |=
      test__adddf3(0x4000000000000002, 0xbff0000000000001, 0x3ff0000000000003);
  status |=
      test__adddf3(0x4000000000000002, 0xbff0000000000003, 0x3ff0000000000001);
  status |=
      test__adddf3(0x4000000000000004, 0xc000000000000003, 0x3cc0000000000000);
  status |=
      test__adddf3(0x4008000000000000, 0x4008000000000000, 0x4018000000000000);
  status |=
      test__adddf3(0x400fffffffffffff, 0x3cafffffffffffff, 0x400fffffffffffff);
  status |=
      test__adddf3(0x400fffffffffffff, 0x3cb0000000000000, 0x4010000000000000);
  status |=
      test__adddf3(0x400fffffffffffff, 0xc00ffffffffffffe, 0x3cc0000000000000);
  status |=
      test__adddf3(0x400fffffffffffff, 0xc010000000000002, 0xbce4000000000000);
  status |=
      test__adddf3(0x4010000000000001, 0xc00fffffffffffff, 0x3cd8000000000000);
  status |=
      test__adddf3(0x4014000000000000, 0x0000000000000000, 0x4014000000000000);
  status |=
      test__adddf3(0x4014000000000000, 0x8000000000000000, 0x4014000000000000);
  status |=
      test__adddf3(0x4014000000000000, 0xbff0000000000000, 0x4010000000000000);
  status |=
      test__adddf3(0x4014000000000000, 0xc014000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x7fb0000000000001, 0xffafffffffffffff, 0x7c78000000000000);
  status |=
      test__adddf3(0x7fcfffffffffffff, 0xffcffffffffffffe, 0x7c80000000000000);
  status |=
      test__adddf3(0x7fcfffffffffffff, 0xffd0000000000002, 0xfca4000000000000);
  status |=
      test__adddf3(0x7fd0000000000000, 0x7fd0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x7fd0000000000000, 0xffcfffffffffffff, 0x7c80000000000000);
  status |=
      test__adddf3(0x7fd0000000000000, 0xffd0000000000001, 0xfc90000000000000);
  status |=
      test__adddf3(0x7fd0000000000001, 0x7fd0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x7fd0000000000001, 0xffe0000000000001, 0xffd0000000000001);
  status |=
      test__adddf3(0x7fd0000000000002, 0xffc0000000000003, 0x7fc0000000000001);
  status |=
      test__adddf3(0x7fd0000000000004, 0xffd0000000000003, 0x7c90000000000000);
  status |=
      test__adddf3(0x7fdffffffffffffe, 0x7fdffffffffffffe, 0x7feffffffffffffe);
  status |=
      test__adddf3(0x7fdffffffffffffe, 0x7fdfffffffffffff, 0x7feffffffffffffe);
  status |=
      test__adddf3(0x7fdfffffffffffff, 0x3ff0000000000000, 0x7fdfffffffffffff);
  status |=
      test__adddf3(0x7fdfffffffffffff, 0x7fe0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7fdfffffffffffff, 0xbff0000000000000, 0x7fdfffffffffffff);
  status |=
      test__adddf3(0x7fdfffffffffffff, 0xffe0000000000000, 0xfc90000000000000);
  status |=
      test__adddf3(0x7fe0000000000000, 0x3ff0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x7fe0000000000000, 0x7fe0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7fe0000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7fe0000000000000, 0xbff0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x7fe0000000000000, 0xffe0000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x7fe0000000000000, 0xfff0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0x7fe0000000000001, 0x7fe0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7fe0000000000001, 0xffe0000000000000, 0x7ca0000000000000);
  status |=
      test__adddf3(0x7fe0000000000001, 0xffe0000000000002, 0xfca0000000000000);
  status |=
      test__adddf3(0x7fe0000000000002, 0xffd0000000000001, 0x7fd0000000000003);
  status |=
      test__adddf3(0x7feffffffffffffe, 0x3ff0000000000000, 0x7feffffffffffffe);
  status |=
      test__adddf3(0x7feffffffffffffe, 0x7feffffffffffffe, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7feffffffffffffe, 0x7fefffffffffffff, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7feffffffffffffe, 0xbff0000000000000, 0x7feffffffffffffe);
  status |=
      test__adddf3(0x7feffffffffffffe, 0xffefffffffffffff, 0xfca0000000000000);
  status |=
      test__adddf3(0x7fefffffffffffff, 0x3ff0000000000000, 0x7fefffffffffffff);
  status |=
      test__adddf3(0x7fefffffffffffff, 0x8000000000000001, 0x7fefffffffffffff);
  status |=
      test__adddf3(0x7fefffffffffffff, 0xbff0000000000000, 0x7fefffffffffffff);
  status |=
      test__adddf3(0x7fefffffffffffff, 0xffefffffffffffff, 0x0000000000000000);
  status |=
      test__adddf3(0x7ff0000000000000, 0x0000000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7ff0000000000000, 0x000fffffffffffff, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7ff0000000000000, 0x7fe0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7ff0000000000000, 0x8000000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7ff0000000000000, 0x800fffffffffffff, 0x7ff0000000000000);
  status |=
      test__adddf3(0x7ff0000000000000, 0xffe0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x8000000000000000, 0x0000000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x8000000000000000, 0x000fffffffffffff, 0x000fffffffffffff);
  status |=
      test__adddf3(0x8000000000000000, 0x7fe0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x8000000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x8000000000000000, 0x8000000000000000, 0x8000000000000000);
  status |=
      test__adddf3(0x8000000000000000, 0x800fffffffffffff, 0x800fffffffffffff);
  status |=
      test__adddf3(0x8000000000000000, 0x8010000000000000, 0x8010000000000000);
  status |=
      test__adddf3(0x8000000000000000, 0xbff0000000000000, 0xbff0000000000000);
  status |=
      test__adddf3(0x8000000000000000, 0xfff0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0x8000000000000001, 0x0000000000000001, 0x0000000000000000);
  status |=
      test__adddf3(0x8000000000000001, 0x8000000000000001, 0x8000000000000002);
  status |=
      test__adddf3(0x8000000000000001, 0xbfefffffffffffff, 0xbfefffffffffffff);
  status |=
      test__adddf3(0x8000000000000001, 0xbff0000000000000, 0xbff0000000000000);
  status |=
      test__adddf3(0x8000000000000001, 0xbffffffffffffffe, 0xbffffffffffffffe);
  status |=
      test__adddf3(0x8000000000000001, 0xbfffffffffffffff, 0xbfffffffffffffff);
  status |=
      test__adddf3(0x8000000000000001, 0xffdfffffffffffff, 0xffdfffffffffffff);
  status |=
      test__adddf3(0x8000000000000001, 0xffe0000000000000, 0xffe0000000000000);
  status |=
      test__adddf3(0x8000000000000001, 0xffeffffffffffffe, 0xffeffffffffffffe);
  status |=
      test__adddf3(0x8000000000000001, 0xffefffffffffffff, 0xffefffffffffffff);
  status |=
      test__adddf3(0x8000000000000002, 0x0000000000000001, 0x8000000000000001);
  status |=
      test__adddf3(0x8000000000000003, 0x0000000000000000, 0x8000000000000003);
  status |=
      test__adddf3(0x8000000000000003, 0x0000000000000002, 0x8000000000000001);
  status |=
      test__adddf3(0x8000000000000003, 0x4008000000000000, 0x4008000000000000);
  status |=
      test__adddf3(0x8000000000000003, 0x7fe0000000000000, 0x7fe0000000000000);
  status |=
      test__adddf3(0x8000000000000003, 0x7ff0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0x8000000000000003, 0x8000000000000000, 0x8000000000000003);
  status |=
      test__adddf3(0x8000000000000003, 0xfff0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0x8000000000000004, 0x8000000000000004, 0x8000000000000008);
  status |=
      test__adddf3(0x800ffffffffffffd, 0x000ffffffffffffe, 0x0000000000000001);
  status |=
      test__adddf3(0x800fffffffffffff, 0x000ffffffffffffe, 0x8000000000000001);
  status |=
      test__adddf3(0x800fffffffffffff, 0x000fffffffffffff, 0x0000000000000000);
  status |=
      test__adddf3(0x800fffffffffffff, 0x0010000000000000, 0x0000000000000001);
  status |=
      test__adddf3(0x800fffffffffffff, 0x800fffffffffffff, 0x801ffffffffffffe);
  status |=
      test__adddf3(0x8010000000000000, 0x0000000000000000, 0x8010000000000000);
  status |=
      test__adddf3(0x8010000000000000, 0x0010000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0x8010000000000001, 0x0010000000000000, 0x8000000000000001);
  status |=
      test__adddf3(0x8010000000000001, 0x0010000000000002, 0x0000000000000001);
  status |=
      test__adddf3(0x801fffffffffffff, 0x0020000000000000, 0x0000000000000001);
  status |=
      test__adddf3(0x801fffffffffffff, 0x0020000000000002, 0x0000000000000005);
  status |=
      test__adddf3(0x801fffffffffffff, 0x0020000000000004, 0x0000000000000009);
  status |=
      test__adddf3(0x8020000000000000, 0x001fffffffffffff, 0x8000000000000001);
  status |=
      test__adddf3(0x8020000000000001, 0x0010000000000001, 0x8010000000000001);
  status |=
      test__adddf3(0x8020000000000001, 0x001fffffffffffff, 0x8000000000000003);
  status |=
      test__adddf3(0x8020000000000002, 0x0010000000000001, 0x8010000000000003);
  status |=
      test__adddf3(0x802fffffffffffff, 0x0030000000000000, 0x0000000000000002);
  status |=
      test__adddf3(0x8030000000000000, 0x002fffffffffffff, 0x8000000000000002);
  status |=
      test__adddf3(0x8030000000000001, 0x002fffffffffffff, 0x8000000000000006);
  status |=
      test__adddf3(0x8030000000000002, 0x0020000000000003, 0x8020000000000001);
  status |=
      test__adddf3(0xbff0000000000000, 0x8000000000000000, 0xbff0000000000000);
  status |=
      test__adddf3(0xbff0000000000000, 0xbff0000000000003, 0xc000000000000002);
  status |=
      test__adddf3(0xbff0000000000001, 0x3ff0000000000000, 0xbcb0000000000000);
  status |=
      test__adddf3(0xbff0000000000001, 0x3ff0000000000002, 0x3cb0000000000000);
  status |=
      test__adddf3(0xbff0000000000001, 0xbff0000000000000, 0xc000000000000000);
  status |=
      test__adddf3(0xbffffffffffffffc, 0x3ffffffffffffffd, 0x3cb0000000000000);
  status |=
      test__adddf3(0xbfffffffffffffff, 0x0000000000000001, 0xbfffffffffffffff);
  status |=
      test__adddf3(0xbfffffffffffffff, 0x4000000000000000, 0x3cb0000000000000);
  status |=
      test__adddf3(0xc000000000000000, 0x3fffffffffffffff, 0xbcb0000000000000);
  status |=
      test__adddf3(0xc000000000000000, 0x4000000000000001, 0x3cc0000000000000);
  status |=
      test__adddf3(0xc000000000000000, 0xc000000000000001, 0xc010000000000000);
  status |=
      test__adddf3(0xc000000000000001, 0x3ff0000000000001, 0xbff0000000000001);
  status |=
      test__adddf3(0xc000000000000001, 0xc000000000000002, 0xc010000000000002);
  status |=
      test__adddf3(0xc000000000000002, 0x3ff0000000000001, 0xbff0000000000003);
  status |=
      test__adddf3(0xc000000000000002, 0x3ff0000000000003, 0xbff0000000000001);
  status |=
      test__adddf3(0xc000000000000004, 0x4000000000000003, 0xbcc0000000000000);
  status |=
      test__adddf3(0xc008000000000000, 0x4008000000000000, 0x0000000000000000);
  status |=
      test__adddf3(0xc00fffffffffffff, 0x400ffffffffffffe, 0xbcc0000000000000);
  status |=
      test__adddf3(0xc00fffffffffffff, 0x4010000000000002, 0x3ce4000000000000);
  status |=
      test__adddf3(0xc00fffffffffffff, 0xbcafffffffffffff, 0xc00fffffffffffff);
  status |=
      test__adddf3(0xc00fffffffffffff, 0xbcb0000000000000, 0xc010000000000000);
  status |=
      test__adddf3(0xc010000000000001, 0x400fffffffffffff, 0xbcd8000000000000);
  status |=
      test__adddf3(0xffb0000000000001, 0x7fafffffffffffff, 0xfc78000000000000);
  status |=
      test__adddf3(0xffcfffffffffffff, 0x7fcffffffffffffe, 0xfc80000000000000);
  status |=
      test__adddf3(0xffcfffffffffffff, 0x7fd0000000000002, 0x7ca4000000000000);
  status |=
      test__adddf3(0xffd0000000000000, 0x7fcfffffffffffff, 0xfc80000000000000);
  status |=
      test__adddf3(0xffd0000000000000, 0x7fd0000000000001, 0x7c90000000000000);
  status |=
      test__adddf3(0xffd0000000000001, 0x7fe0000000000001, 0x7fd0000000000001);
  status |=
      test__adddf3(0xffd0000000000001, 0xffd0000000000000, 0xffe0000000000000);
  status |=
      test__adddf3(0xffd0000000000002, 0x7fc0000000000003, 0xffc0000000000001);
  status |=
      test__adddf3(0xffd0000000000004, 0x7fd0000000000003, 0xfc90000000000000);
  status |=
      test__adddf3(0xffdffffffffffffe, 0x7fdffffffffffffe, 0x0000000000000000);
  status |=
      test__adddf3(0xffdffffffffffffe, 0xffdffffffffffffe, 0xffeffffffffffffe);
  status |=
      test__adddf3(0xffdffffffffffffe, 0xffdfffffffffffff, 0xffeffffffffffffe);
  status |=
      test__adddf3(0xffdfffffffffffff, 0x3ff0000000000000, 0xffdfffffffffffff);
  status |=
      test__adddf3(0xffdfffffffffffff, 0x7fe0000000000000, 0x7c90000000000000);
  status |=
      test__adddf3(0xffdfffffffffffff, 0xbff0000000000000, 0xffdfffffffffffff);
  status |=
      test__adddf3(0xffdfffffffffffff, 0xffe0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xffe0000000000000, 0x0000000000000000, 0xffe0000000000000);
  status |=
      test__adddf3(0xffe0000000000000, 0x3ff0000000000000, 0xffe0000000000000);
  status |=
      test__adddf3(0xffe0000000000000, 0x7ff0000000000000, 0x7ff0000000000000);
  status |=
      test__adddf3(0xffe0000000000000, 0x8000000000000000, 0xffe0000000000000);
  status |=
      test__adddf3(0xffe0000000000000, 0xbff0000000000000, 0xffe0000000000000);
  status |=
      test__adddf3(0xffe0000000000000, 0xffe0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xffe0000000000000, 0xfff0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xffe0000000000001, 0x7fe0000000000000, 0xfca0000000000000);
  status |=
      test__adddf3(0xffe0000000000001, 0x7fe0000000000002, 0x7ca0000000000000);
  status |=
      test__adddf3(0xffe0000000000001, 0xffe0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xffe0000000000002, 0x7fd0000000000001, 0xffd0000000000003);
  status |=
      test__adddf3(0xffeffffffffffffe, 0x3ff0000000000000, 0xffeffffffffffffe);
  status |=
      test__adddf3(0xffeffffffffffffe, 0x7fefffffffffffff, 0x7ca0000000000000);
  status |=
      test__adddf3(0xffeffffffffffffe, 0xbff0000000000000, 0xffeffffffffffffe);
  status |=
      test__adddf3(0xffeffffffffffffe, 0xffeffffffffffffe, 0xfff0000000000000);
  status |=
      test__adddf3(0xffeffffffffffffe, 0xffefffffffffffff, 0xfff0000000000000);
  status |=
      test__adddf3(0xffefffffffffffff, 0x0000000000000001, 0xffefffffffffffff);
  status |=
      test__adddf3(0xffefffffffffffff, 0x3ff0000000000000, 0xffefffffffffffff);
  status |=
      test__adddf3(0xffefffffffffffff, 0xbff0000000000000, 0xffefffffffffffff);
  status |=
      test__adddf3(0xfff0000000000000, 0x0000000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xfff0000000000000, 0x000fffffffffffff, 0xfff0000000000000);
  status |=
      test__adddf3(0xfff0000000000000, 0x7fe0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xfff0000000000000, 0x8000000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xfff0000000000000, 0x800fffffffffffff, 0xfff0000000000000);
  status |=
      test__adddf3(0xfff0000000000000, 0xffe0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0xfff0000000000000, 0xfff0000000000000, 0xfff0000000000000);
  status |=
      test__adddf3(0x3de3a83a83a83a83, 0xbff0000000000000, 0xbfefffffffec57c5);
  status |=
      test__adddf3(0x0000000007ffffff, 0x0010000000010000, 0x001000000800ffff);
  status |=
      test__adddf3(0x001effffffffffff, 0x0000000000400000, 0x001f0000003fffff);
  status |=
      test__adddf3(0x80000000000003ff, 0x801ffffbffffffff, 0x801ffffc000003fe);
  status |=
      test__adddf3(0x80003fffffffffff, 0x8010000000100000, 0x80104000000fffff);

  // Test that the result of an operation is a NaN at all when it should be.
  //
  // In most configurations these tests' results are checked compared using
  // compareResultD, so we set all the answers to the canonical NaN
  // 0x7ff8000000000000, which causes compareResultF to accept any NaN
  // encoding. We also use the same value as the input NaN in tests that have
  // one, so that even in EXPECT_EXACT_RESULTS mode these tests should pass,
  // because 0x7ff8000000000000 is still the exact expected NaN.
  status |=
      test__adddf3(0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000);
  status |=
      test__adddf3(0xfff0000000000000, 0x7ff0000000000000, 0x7ff8000000000000);
  status |=
      test__adddf3(0x3ff0000000000000, 0x7ff8000000000000, 0x7ff8000000000000);
  status |=
      test__adddf3(0x7ff8000000000000, 0x3ff0000000000000, 0x7ff8000000000000);
  status |=
      test__adddf3(0x7ff8000000000000, 0x7ff8000000000000, 0x7ff8000000000000);

#ifdef ARM_NAN_HANDLING
  // Tests specific to the NaN handling of Arm hardware, mimicked by
  // arm/adddf3.S:
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
  status |=
      test__adddf3(0x0000000000000000, 0x7ff3758244400801, 0x7ffb758244400801);
  status |=
      test__adddf3(0x0000000000000000, 0x7fff44d3f65148af, 0x7fff44d3f65148af);
  status |=
      test__adddf3(0x0000000000000001, 0x7ff48607b4b37057, 0x7ffc8607b4b37057);
  status |=
      test__adddf3(0x0000000000000001, 0x7ff855f2d435b33d, 0x7ff855f2d435b33d);
  status |=
      test__adddf3(0x000fffffffffffff, 0x7ff169269a674e13, 0x7ff969269a674e13);
  status |=
      test__adddf3(0x000fffffffffffff, 0x7ffc80978b2ef0da, 0x7ffc80978b2ef0da);
  status |=
      test__adddf3(0x3ff0000000000000, 0x7ff3458ad034593d, 0x7ffb458ad034593d);
  status |=
      test__adddf3(0x3ff0000000000000, 0x7ffdd8bb98c9f13a, 0x7ffdd8bb98c9f13a);
  status |=
      test__adddf3(0x7fefffffffffffff, 0x7ff79a8b96250a98, 0x7fff9a8b96250a98);
  status |=
      test__adddf3(0x7fefffffffffffff, 0x7ffdcc675b63bb94, 0x7ffdcc675b63bb94);
  status |=
      test__adddf3(0x7ff0000000000000, 0x7ff018cfaf4d0fff, 0x7ff818cfaf4d0fff);
  status |=
      test__adddf3(0x7ff0000000000000, 0x7ff83ad1ab4dfd24, 0x7ff83ad1ab4dfd24);
  status |=
      test__adddf3(0x7ff48ce6c0cdd5ac, 0x0000000000000000, 0x7ffc8ce6c0cdd5ac);
  status |=
      test__adddf3(0x7ff08a34f3d5385b, 0x0000000000000001, 0x7ff88a34f3d5385b);
  status |=
      test__adddf3(0x7ff0a264c1c96281, 0x000fffffffffffff, 0x7ff8a264c1c96281);
  status |=
      test__adddf3(0x7ff77ce629e61f0e, 0x3ff0000000000000, 0x7fff7ce629e61f0e);
  status |=
      test__adddf3(0x7ff715e2d147fd76, 0x7fefffffffffffff, 0x7fff15e2d147fd76);
  status |=
      test__adddf3(0x7ff689a2031f1781, 0x7ff0000000000000, 0x7ffe89a2031f1781);
  status |=
      test__adddf3(0x7ff5dfb4a0c8cd05, 0x7ff11c1fe9793a33, 0x7ffddfb4a0c8cd05);
  status |=
      test__adddf3(0x7ff5826283ffb5d7, 0x7fff609b83884e81, 0x7ffd826283ffb5d7);
  status |=
      test__adddf3(0x7ff7cb03f2e61d42, 0x8000000000000000, 0x7fffcb03f2e61d42);
  status |=
      test__adddf3(0x7ff2adc8dfe72c96, 0x8000000000000001, 0x7ffaadc8dfe72c96);
  status |=
      test__adddf3(0x7ff4fc0bacc707f2, 0x800fffffffffffff, 0x7ffcfc0bacc707f2);
  status |=
      test__adddf3(0x7ff76248c8c9a619, 0xbff0000000000000, 0x7fff6248c8c9a619);
  status |=
      test__adddf3(0x7ff367972fce131b, 0xffefffffffffffff, 0x7ffb67972fce131b);
  status |=
      test__adddf3(0x7ff188f5ac284e92, 0xfff0000000000000, 0x7ff988f5ac284e92);
  status |=
      test__adddf3(0x7ffed4c22e4e569d, 0x0000000000000000, 0x7ffed4c22e4e569d);
  status |=
      test__adddf3(0x7ffe95105fa3f339, 0x0000000000000001, 0x7ffe95105fa3f339);
  status |=
      test__adddf3(0x7ffb8d33dbb9ecfb, 0x000fffffffffffff, 0x7ffb8d33dbb9ecfb);
  status |=
      test__adddf3(0x7ff874e41dc63e07, 0x3ff0000000000000, 0x7ff874e41dc63e07);
  status |=
      test__adddf3(0x7ffe27594515ecdf, 0x7fefffffffffffff, 0x7ffe27594515ecdf);
  status |=
      test__adddf3(0x7ffeac86d5c69bdf, 0x7ff0000000000000, 0x7ffeac86d5c69bdf);
  status |=
      test__adddf3(0x7ff97d657b99f76f, 0x7ff7e4149862a796, 0x7fffe4149862a796);
  status |=
      test__adddf3(0x7ffad17c6aa33fad, 0x7ffd898893ad4d28, 0x7ffad17c6aa33fad);
  status |=
      test__adddf3(0x7ff96e04e9c3d173, 0x8000000000000000, 0x7ff96e04e9c3d173);
  status |=
      test__adddf3(0x7ffec01ad8da3abb, 0x8000000000000001, 0x7ffec01ad8da3abb);
  status |=
      test__adddf3(0x7ffd1d565c495941, 0x800fffffffffffff, 0x7ffd1d565c495941);
  status |=
      test__adddf3(0x7ffe3d24f1e474a7, 0xbff0000000000000, 0x7ffe3d24f1e474a7);
  status |=
      test__adddf3(0x7ffc206f2bb8c8ce, 0xffefffffffffffff, 0x7ffc206f2bb8c8ce);
  status |=
      test__adddf3(0x7ff93efdecfb7d3b, 0xfff0000000000000, 0x7ff93efdecfb7d3b);
  status |=
      test__adddf3(0x8000000000000000, 0x7ff2ee725d143ac5, 0x7ffaee725d143ac5);
  status |=
      test__adddf3(0x8000000000000000, 0x7ffbba26e5c5fe98, 0x7ffbba26e5c5fe98);
  status |=
      test__adddf3(0x8000000000000001, 0x7ff7818a1cd26df9, 0x7fff818a1cd26df9);
  status |=
      test__adddf3(0x8000000000000001, 0x7ffaee6cc63b5292, 0x7ffaee6cc63b5292);
  status |=
      test__adddf3(0x800fffffffffffff, 0x7ff401096edaf79d, 0x7ffc01096edaf79d);
  status |=
      test__adddf3(0x800fffffffffffff, 0x7ffbf1778c7a2e59, 0x7ffbf1778c7a2e59);
  status |=
      test__adddf3(0xbff0000000000000, 0x7ff2e8fb0201c496, 0x7ffae8fb0201c496);
  status |=
      test__adddf3(0xbff0000000000000, 0x7ffcb6a5adb2e154, 0x7ffcb6a5adb2e154);
  status |=
      test__adddf3(0xffefffffffffffff, 0x7ff1ea1bfc15d71d, 0x7ff9ea1bfc15d71d);
  status |=
      test__adddf3(0xffefffffffffffff, 0x7ffae0766e21efc0, 0x7ffae0766e21efc0);
  status |=
      test__adddf3(0xfff0000000000000, 0x7ff3b364cffbdfe6, 0x7ffbb364cffbdfe6);
  status |=
      test__adddf3(0xfff0000000000000, 0x7ffd0d3223334ae3, 0x7ffd0d3223334ae3);

#endif // ARM_NAN_HANDLING

  return status;
}
