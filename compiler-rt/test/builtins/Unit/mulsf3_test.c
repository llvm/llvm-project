// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_mulsf3

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// By default this test uses compareResultF to check the returned floats, which
// accepts any returned NaN if the expected result is the canonical NaN value
// 0x7fc00000. For the Arm optimized FP implementation, which commits to a more
// detailed handling of NaNs, we tighten up the check and include some extra
// test cases specific to that NaN policy.
#if (__arm__ && !(__thumb__ && !__thumb2__)) && COMPILER_RT_ARM_OPTIMIZED_FP
#  define EXPECT_EXACT_RESULTS
#  define ARM_NAN_HANDLING
#endif

// Returns: a * b
COMPILER_RT_ABI float __mulsf3(float a, float b);

int test__mulsf3(uint32_t a_rep, uint32_t b_rep, uint32_t expected_rep) {
  float a = fromRep32(a_rep), b = fromRep32(b_rep);
  float x = __mulsf3(a, b);
#ifdef EXPECT_EXACT_RESULTS
  int ret = toRep32(x) == expected_rep;
#else
  int ret = compareResultF(x, expected_rep);
#endif

  if (ret) {
    printf("error in test__mulsf3(%08" PRIx32 ", %08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           a_rep, b_rep, toRep32(x), expected_rep);
  }
  return ret;
}

int main(void) {
  int status = 0;

  status |= test__mulsf3(0x00000000, 0x00000000, 0x00000000);
  status |= test__mulsf3(0x00000000, 0x007fffff, 0x00000000);
  status |= test__mulsf3(0x00000000, 0x00ffffff, 0x00000000);
  status |= test__mulsf3(0x00000000, 0x3f800000, 0x00000000);
  status |= test__mulsf3(0x00000000, 0x7effffff, 0x00000000);
  status |= test__mulsf3(0x00000000, 0x80000000, 0x80000000);
  status |= test__mulsf3(0x00000000, 0x80000002, 0x80000000);
  status |= test__mulsf3(0x00000000, 0x807fffff, 0x80000000);
  status |= test__mulsf3(0x00000000, 0x80800001, 0x80000000);
  status |= test__mulsf3(0x00000000, 0x81000000, 0x80000000);
  status |= test__mulsf3(0x00000000, 0xc0400000, 0x80000000);
  status |= test__mulsf3(0x00000000, 0xfe7fffff, 0x80000000);
  status |= test__mulsf3(0x00000000, 0xff000000, 0x80000000);
  status |= test__mulsf3(0x00000000, 0xff7fffff, 0x80000000);
  status |= test__mulsf3(0x00000001, 0x00000000, 0x00000000);
  status |= test__mulsf3(0x00000001, 0x00000001, 0x00000000);
  status |= test__mulsf3(0x00000001, 0x3f000000, 0x00000000);
  status |= test__mulsf3(0x00000001, 0x3f7fffff, 0x00000001);
  status |= test__mulsf3(0x00000001, 0x3f800000, 0x00000001);
  status |= test__mulsf3(0x00000001, 0x40000000, 0x00000002);
  status |= test__mulsf3(0x00000001, 0x7f800000, 0x7f800000);
  status |= test__mulsf3(0x00000001, 0xbf7fffff, 0x80000001);
  status |= test__mulsf3(0x00000006, 0x3f000000, 0x00000003);
  status |= test__mulsf3(0x00000006, 0xbf000000, 0x80000003);
  status |= test__mulsf3(0x00000008, 0x3e000000, 0x00000001);
  status |= test__mulsf3(0x007ffff7, 0x81000003, 0x80000000);
  status |= test__mulsf3(0x007ffff8, 0x3f800001, 0x007ffff9);
  status |= test__mulsf3(0x007ffff8, 0x3f800008, 0x00800000);
  status |= test__mulsf3(0x007ffff8, 0xbf800001, 0x807ffff9);
  status |= test__mulsf3(0x007ffff8, 0xbf800008, 0x80800000);
  status |= test__mulsf3(0x007ffffc, 0x40000000, 0x00fffff8);
  status |= test__mulsf3(0x007ffffe, 0x3f7ffffc, 0x007ffffc);
  status |= test__mulsf3(0x007ffffe, 0x3f800001, 0x007fffff);
  status |= test__mulsf3(0x007ffffe, 0xbf800001, 0x807fffff);
  status |= test__mulsf3(0x007fffff, 0x007ffffe, 0x00000000);
  status |= test__mulsf3(0x007fffff, 0x3f800001, 0x00800000);
  status |= test__mulsf3(0x007fffff, 0x40000000, 0x00fffffe);
  status |= test__mulsf3(0x00800000, 0x00000000, 0x00000000);
  status |= test__mulsf3(0x00800000, 0x00800000, 0x00000000);
  status |= test__mulsf3(0x00800000, 0x3f7ffffe, 0x007fffff);
  status |= test__mulsf3(0x00800000, 0x7f800000, 0x7f800000);
  status |= test__mulsf3(0x00800000, 0x80800000, 0x80000000);
  status |= test__mulsf3(0x00800000, 0xc0000000, 0x81000000);
  status |= test__mulsf3(0x00800001, 0x3f7ffffa, 0x007ffffe);
  status |= test__mulsf3(0x00800001, 0x3f7ffffe, 0x00800000);
  status |= test__mulsf3(0x00800001, 0xc0000000, 0x81000001);
  status |= test__mulsf3(0x00800002, 0x3f7ffffc, 0x00800000);
  status |= test__mulsf3(0x00fffff8, 0x3f000000, 0x007ffffc);
  status |= test__mulsf3(0x00fffffe, 0x3f000000, 0x007fffff);
  status |= test__mulsf3(0x00fffffe, 0xbf000000, 0x807fffff);
  status |= test__mulsf3(0x00ffffff, 0x3f000000, 0x00800000);
  status |= test__mulsf3(0x00ffffff, 0xbf000000, 0x80800000);
  status |= test__mulsf3(0x3f000000, 0x80000001, 0x80000000);
  status |= test__mulsf3(0x3f800000, 0x007ffffd, 0x007ffffd);
  status |= test__mulsf3(0x3f800000, 0x01000003, 0x01000003);
  status |= test__mulsf3(0x3f800000, 0x3f800000, 0x3f800000);
  status |= test__mulsf3(0x3f800000, 0x40000000, 0x40000000);
  status |= test__mulsf3(0x3f800000, 0x80000001, 0x80000001);
  status |= test__mulsf3(0x3f800000, 0x80000009, 0x80000009);
  status |= test__mulsf3(0x3f800001, 0x3f800001, 0x3f800002);
  status |= test__mulsf3(0x3f800001, 0xbf800001, 0xbf800002);
  status |= test__mulsf3(0x3f800001, 0xbf800002, 0xbf800003);
  status |= test__mulsf3(0x3f800002, 0x3f800001, 0x3f800003);
  status |= test__mulsf3(0x3f800002, 0x7f7ffffe, 0x7f800000);
  status |= test__mulsf3(0x3f800001, 0x7f7ffffe, 0x7f800000);
  status |= test__mulsf3(0x40000000, 0x00800000, 0x01000000);
  status |= test__mulsf3(0x40000000, 0x00800001, 0x01000001);
  status |= test__mulsf3(0x40000000, 0x3f800000, 0x40000000);
  status |= test__mulsf3(0x40000000, 0x40400000, 0x40c00000);
  status |= test__mulsf3(0x40000000, 0x7e800000, 0x7f000000);
  status |= test__mulsf3(0x40000000, 0x7effffff, 0x7f7fffff);
  status |= test__mulsf3(0x40000000, 0x807ffffd, 0x80fffffa);
  status |= test__mulsf3(0x40000000, 0x80800003, 0x81000003);
  status |= test__mulsf3(0x40000000, 0x80800005, 0x81000005);
  status |= test__mulsf3(0x40000000, 0xbf800000, 0xc0000000);
  status |= test__mulsf3(0x40000000, 0xfe7ffffd, 0xfefffffd);
  status |= test__mulsf3(0x40000000, 0xfe800003, 0xff000003);
  status |= test__mulsf3(0x403fffff, 0x3f7ffffd, 0x403ffffd);
  status |= test__mulsf3(0x403fffff, 0x3f7ffffe, 0x403ffffe);
  status |= test__mulsf3(0x403fffff, 0x3f7fffff, 0x403ffffe);
  status |= test__mulsf3(0x403fffff, 0xbf7ffffd, 0xc03ffffd);
  status |= test__mulsf3(0x40400000, 0x00000002, 0x00000006);
  status |= test__mulsf3(0x40400000, 0x40000000, 0x40c00000);
  status |= test__mulsf3(0x40400000, 0x40400000, 0x41100000);
  status |= test__mulsf3(0x40400000, 0xc0000000, 0xc0c00000);
  status |= test__mulsf3(0x40400001, 0x3f800001, 0x40400003);
  status |= test__mulsf3(0x40400001, 0x3f800003, 0x40400006);
  status |= test__mulsf3(0x40400001, 0xbf800003, 0xc0400006);
  status |= test__mulsf3(0x40800000, 0x00000002, 0x00000008);
  status |= test__mulsf3(0x40800000, 0x7e7fffff, 0x7f7fffff);
  status |= test__mulsf3(0x40800000, 0xfe7fffff, 0xff7fffff);
  status |= test__mulsf3(0x409fffff, 0x3f7fffff, 0x409ffffe);
  status |= test__mulsf3(0x40a00000, 0x00000000, 0x00000000);
  status |= test__mulsf3(0x40a00000, 0x7f800000, 0x7f800000);
  status |= test__mulsf3(0x40a00001, 0x3f800001, 0x40a00002);
  status |= test__mulsf3(0x40dfffff, 0x3f7ffffc, 0x40dffffc);
  status |= test__mulsf3(0x40dfffff, 0x3f7fffff, 0x40dffffe);
  status |= test__mulsf3(0x40e00000, 0x80000000, 0x80000000);
  status |= test__mulsf3(0x40e00000, 0xff800000, 0xff800000);
  status |= test__mulsf3(0x40e00001, 0x3f800001, 0x40e00003);
  status |= test__mulsf3(0x7e7ffffd, 0x40800000, 0x7f7ffffd);
  status |= test__mulsf3(0x7e7ffffd, 0xc0800000, 0xff7ffffd);
  status |= test__mulsf3(0x7e800000, 0xc0000000, 0xff000000);
  status |= test__mulsf3(0x7efffffd, 0xc0000008, 0xff800000);
  status |= test__mulsf3(0x7effffff, 0xc0000000, 0xff7fffff);
  status |= test__mulsf3(0x7f000000, 0x00000000, 0x00000000);
  status |= test__mulsf3(0x7f000000, 0x40000000, 0x7f800000);
  status |= test__mulsf3(0x7f000000, 0x7f000000, 0x7f800000);
  status |= test__mulsf3(0x7f000000, 0x7f7ffffe, 0x7f800000);
  status |= test__mulsf3(0x7f000000, 0x7f800000, 0x7f800000);
  status |= test__mulsf3(0x7f000000, 0xfe800000, 0xff800000);
  status |= test__mulsf3(0x7f000000, 0xfe800004, 0xff800000);
  status |= test__mulsf3(0x7f000000, 0xff000000, 0xff800000);
  status |= test__mulsf3(0x7f000009, 0x7f7ffffa, 0x7f800000);
  status |= test__mulsf3(0x7f000009, 0xc0c00002, 0xff800000);
  status |= test__mulsf3(0x7f7fffff, 0x00000000, 0x00000000);
  status |= test__mulsf3(0x7f800000, 0x007fffff, 0x7f800000);
  status |= test__mulsf3(0x7f800000, 0x00ffffff, 0x7f800000);
  status |= test__mulsf3(0x7f800000, 0x3f800000, 0x7f800000);
  status |= test__mulsf3(0x7f800000, 0x7effffff, 0x7f800000);
  status |= test__mulsf3(0x7f800000, 0x7f800000, 0x7f800000);
  status |= test__mulsf3(0x7f800000, 0x80000002, 0xff800000);
  status |= test__mulsf3(0x7f800000, 0x807fffff, 0xff800000);
  status |= test__mulsf3(0x7f800000, 0x80800001, 0xff800000);
  status |= test__mulsf3(0x7f800000, 0x81000000, 0xff800000);
  status |= test__mulsf3(0x7f800000, 0xc0400000, 0xff800000);
  status |= test__mulsf3(0x7f800000, 0xff000000, 0xff800000);
  status |= test__mulsf3(0x7f800000, 0xff7fffff, 0xff800000);
  status |= test__mulsf3(0x7f800000, 0xff800000, 0xff800000);
  status |= test__mulsf3(0x80000000, 0x00000000, 0x80000000);
  status |= test__mulsf3(0x80000000, 0x40c00000, 0x80000000);
  status |= test__mulsf3(0x80000000, 0x7f7fffff, 0x80000000);
  status |= test__mulsf3(0x80000000, 0x80000000, 0x00000000);
  status |= test__mulsf3(0x80000000, 0x80000004, 0x00000000);
  status |= test__mulsf3(0x80000000, 0x80800000, 0x00000000);
  status |= test__mulsf3(0x80000000, 0xc1000000, 0x00000000);
  status |= test__mulsf3(0x80000000, 0xfe800000, 0x00000000);
  status |= test__mulsf3(0x80000001, 0x00000001, 0x80000000);
  status |= test__mulsf3(0x80000001, 0x40a00000, 0x80000005);
  status |= test__mulsf3(0x80000002, 0x3f800000, 0x80000002);
  status |= test__mulsf3(0x80000003, 0x00000000, 0x80000000);
  status |= test__mulsf3(0x80000003, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0x80000004, 0xbf800000, 0x00000004);
  status |= test__mulsf3(0x80000008, 0x3e000000, 0x80000001);
  status |= test__mulsf3(0x807ffff7, 0x01000003, 0x80000000);
  status |= test__mulsf3(0x807ffff7, 0x3f800001, 0x807ffff8);
  status |= test__mulsf3(0x807ffffd, 0xc0000000, 0x00fffffa);
  status |= test__mulsf3(0x807fffff, 0x00000000, 0x80000000);
  status |= test__mulsf3(0x807fffff, 0x3f800001, 0x80800000);
  status |= test__mulsf3(0x807fffff, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0x807fffff, 0x80000000, 0x00000000);
  status |= test__mulsf3(0x807fffff, 0x807ffffe, 0x00000000);
  status |= test__mulsf3(0x807fffff, 0xbf800000, 0x007fffff);
  status |= test__mulsf3(0x807fffff, 0xff800000, 0x7f800000);
  status |= test__mulsf3(0x80800000, 0x00800000, 0x80000000);
  status |= test__mulsf3(0x80800000, 0x80800000, 0x00000000);
  status |= test__mulsf3(0x80800001, 0x00000000, 0x80000000);
  status |= test__mulsf3(0x80800001, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0x80800001, 0xbf800000, 0x00800001);
  status |= test__mulsf3(0x80fffffc, 0x3f000000, 0x807ffffe);
  status |= test__mulsf3(0x80fffffc, 0xbf000000, 0x007ffffe);
  status |= test__mulsf3(0x80fffffe, 0x3f800000, 0x80fffffe);
  status |= test__mulsf3(0x80ffffff, 0x80000000, 0x00000000);
  status |= test__mulsf3(0x80ffffff, 0xff800000, 0x7f800000);
  status |= test__mulsf3(0x81000000, 0x00000000, 0x80000000);
  status |= test__mulsf3(0x81000000, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0xbf7fffff, 0xff7fffff, 0x7f7ffffe);
  status |= test__mulsf3(0xbf800000, 0x00000009, 0x80000009);
  status |= test__mulsf3(0xbf800000, 0x00800009, 0x80800009);
  status |= test__mulsf3(0xbf800000, 0x3f800000, 0xbf800000);
  status |= test__mulsf3(0xbf800000, 0x40000000, 0xc0000000);
  status |= test__mulsf3(0xbf800000, 0xbf800000, 0x3f800000);
  status |= test__mulsf3(0xbf800000, 0xc0000000, 0x40000000);
  status |= test__mulsf3(0xbf800001, 0x3f800001, 0xbf800002);
  status |= test__mulsf3(0xbf800001, 0xbf800001, 0x3f800002);
  status |= test__mulsf3(0xbf800001, 0xbf800002, 0x3f800003);
  status |= test__mulsf3(0xbf800002, 0x3f800001, 0xbf800003);
  status |= test__mulsf3(0xbf800002, 0xbf800001, 0x3f800003);
  status |= test__mulsf3(0xc0000000, 0x00000000, 0x80000000);
  status |= test__mulsf3(0xc0000000, 0x007ffffd, 0x80fffffa);
  status |= test__mulsf3(0xc0000000, 0x00800001, 0x81000001);
  status |= test__mulsf3(0xc0000000, 0x00800005, 0x81000005);
  status |= test__mulsf3(0xc0000000, 0x00800009, 0x81000009);
  status |= test__mulsf3(0xc0000000, 0x40400000, 0xc0c00000);
  status |= test__mulsf3(0xc0000000, 0x7e7fffff, 0xfeffffff);
  status |= test__mulsf3(0xc0000000, 0x7e800001, 0xff000001);
  status |= test__mulsf3(0xc0000000, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0xc0000000, 0xbf800000, 0x40000000);
  status |= test__mulsf3(0xc0000000, 0xc0400000, 0x40c00000);
  status |= test__mulsf3(0xc03ffffe, 0x7f000000, 0xff800000);
  status |= test__mulsf3(0xc03fffff, 0x3f7fffff, 0xc03ffffe);
  status |= test__mulsf3(0xc0400000, 0x40400000, 0xc1100000);
  status |= test__mulsf3(0xc0400000, 0xc0000000, 0x40c00000);
  status |= test__mulsf3(0xc0400000, 0xc0400000, 0x41100000);
  status |= test__mulsf3(0xc0400000, 0xff000000, 0x7f800000);
  status |= test__mulsf3(0xc0400001, 0x3f800001, 0xc0400003);
  status |= test__mulsf3(0xc0800000, 0x7e7fffff, 0xff7fffff);
  status |= test__mulsf3(0xc0800000, 0x80000000, 0x00000000);
  status |= test__mulsf3(0xc0800000, 0xfe7fffff, 0x7f7fffff);
  status |= test__mulsf3(0xc0800000, 0xff800000, 0x7f800000);
  status |= test__mulsf3(0xc09ffffe, 0xff000000, 0x7f800000);
  status |= test__mulsf3(0xc09fffff, 0xbf7fffff, 0x409ffffe);
  status |= test__mulsf3(0xc0a00001, 0xbf800001, 0x40a00002);
  status |= test__mulsf3(0xc0dffff9, 0x7f000000, 0xff800000);
  status |= test__mulsf3(0xc1100000, 0x7f000000, 0xff800000);
  status |= test__mulsf3(0xc1100001, 0xff000000, 0x7f800000);
  status |= test__mulsf3(0xfe7ffff9, 0x7f000000, 0xff800000);
  status |= test__mulsf3(0xfe7ffff9, 0xc07fffff, 0x7f7ffff8);
  status |= test__mulsf3(0xfe7ffffd, 0x40800000, 0xff7ffffd);
  status |= test__mulsf3(0xfe7ffffd, 0xc0800000, 0x7f7ffffd);
  status |= test__mulsf3(0xfe7fffff, 0x00000000, 0x80000000);
  status |= test__mulsf3(0xfe7fffff, 0x40000001, 0xff000000);
  status |= test__mulsf3(0xfe7fffff, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0xfe800000, 0x00000000, 0x80000000);
  status |= test__mulsf3(0xfe800000, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0xfefffff7, 0x7e800001, 0xff800000);
  status |= test__mulsf3(0xfeffffff, 0x3f800001, 0xff000000);
  status |= test__mulsf3(0xfeffffff, 0x80000000, 0x00000000);
  status |= test__mulsf3(0xff000005, 0xff000001, 0x7f800000);
  status |= test__mulsf3(0xff7ffffd, 0x7f000000, 0xff800000);
  status |= test__mulsf3(0xff7ffffd, 0xc0400001, 0x7f800000);
  status |= test__mulsf3(0xff7ffffd, 0xff000001, 0x7f800000);
  status |= test__mulsf3(0xff7fffff, 0x80000000, 0x00000000);
  status |= test__mulsf3(0xff7fffff, 0xff7fffff, 0x7f800000);
  status |= test__mulsf3(0xff7fffff, 0xff800000, 0x7f800000);
  status |= test__mulsf3(0xff800000, 0x40c00000, 0xff800000);
  status |= test__mulsf3(0xff800000, 0x7f800000, 0xff800000);
  status |= test__mulsf3(0xff800000, 0x80000004, 0x7f800000);
  status |= test__mulsf3(0xff800000, 0x80800000, 0x7f800000);
  status |= test__mulsf3(0xff800000, 0xc1000000, 0x7f800000);
  status |= test__mulsf3(0xff800000, 0xfe800000, 0x7f800000);
  status |= test__mulsf3(0xff800000, 0xff800000, 0x7f800000);
  status |= test__mulsf3(0x3089705f, 0x0ef36390, 0x0041558f);
  status |= test__mulsf3(0x3089705f, 0x0e936390, 0x0027907d);
  status |= test__mulsf3(0x3109705f, 0x0ef36390, 0x0082ab1e);
  status |= test__mulsf3(0x3109705f, 0x0e936390, 0x004f20fa);
  status |= test__mulsf3(0x3189705f, 0x0ef36390, 0x0102ab1e);
  status |= test__mulsf3(0x3189705f, 0x0e936390, 0x009e41f5);
  status |= test__mulsf3(0xb089705f, 0x0ef36390, 0x8041558f);
  status |= test__mulsf3(0xb089705f, 0x0e936390, 0x8027907d);
  status |= test__mulsf3(0xb109705f, 0x0ef36390, 0x8082ab1e);
  status |= test__mulsf3(0xb109705f, 0x0e936390, 0x804f20fa);
  status |= test__mulsf3(0xb189705f, 0x0ef36390, 0x8102ab1e);
  status |= test__mulsf3(0xb189705f, 0x0e936390, 0x809e41f5);
  status |= test__mulsf3(0x3089705f, 0x8ef36390, 0x8041558f);
  status |= test__mulsf3(0x3089705f, 0x8e936390, 0x8027907d);
  status |= test__mulsf3(0x3109705f, 0x8ef36390, 0x8082ab1e);
  status |= test__mulsf3(0x3109705f, 0x8e936390, 0x804f20fa);
  status |= test__mulsf3(0x3189705f, 0x8ef36390, 0x8102ab1e);
  status |= test__mulsf3(0x3189705f, 0x8e936390, 0x809e41f5);
  status |= test__mulsf3(0xb089705f, 0x8ef36390, 0x0041558f);
  status |= test__mulsf3(0xb089705f, 0x8e936390, 0x0027907d);
  status |= test__mulsf3(0xb109705f, 0x8ef36390, 0x0082ab1e);
  status |= test__mulsf3(0xb109705f, 0x8e936390, 0x004f20fa);
  status |= test__mulsf3(0xb189705f, 0x8ef36390, 0x0102ab1e);
  status |= test__mulsf3(0xb189705f, 0x8e936390, 0x009e41f5);
  status |= test__mulsf3(0x1f800001, 0x1fc00000, 0x00300000);
  status |= test__mulsf3(0x1f800003, 0x1fc00000, 0x00300001);
  status |= test__mulsf3(0x1f800001, 0x1fc00800, 0x00300200);
  status |= test__mulsf3(0x1f800003, 0x1fc00800, 0x00300201);
  status |= test__mulsf3(0x36e4588a, 0x29b47cbd, 0x2120fd85);
  status |= test__mulsf3(0x3fea3b26, 0x3f400000, 0x3fafac5c);
  status |= test__mulsf3(0x6fea3b26, 0x4f400000, 0x7f800000);
  status |= test__mulsf3(0x20ea3b26, 0x1ec00000, 0x0057d62e);
  status |= test__mulsf3(0x3f8f11bb, 0x3fc00000, 0x3fd69a98);
  status |= test__mulsf3(0x6f8f11bb, 0x4fc00000, 0x7f800000);
  status |= test__mulsf3(0x208f11bb, 0x1f400000, 0x006b4d4c);
  status |= test__mulsf3(0x3f8f11bb, 0x3f800000, 0x3f8f11bb);
  status |= test__mulsf3(0x6f8f11bb, 0x4f800000, 0x7f800000);
  status |= test__mulsf3(0x208f11bb, 0x1f000000, 0x004788de);
  status |= test__mulsf3(0x3f8f11bb, 0x3fd7f48d, 0x3ff1611f);
  status |= test__mulsf3(0x6f8f11bb, 0x4fd7f48d, 0x7f800000);
  status |= test__mulsf3(0x208f11bb, 0x1f57f48d, 0x0078b090);
  status |= test__mulsf3(0x3f8f11bb, 0x3fa80b73, 0x3fbbd412);
  status |= test__mulsf3(0x6f8f11bb, 0x4fa80b73, 0x7f800000);
  status |= test__mulsf3(0x208f11bb, 0x1f280b73, 0x005dea09);
  status |= test__mulsf3(0x3f8f11bb, 0x3f97f48d, 0x3fa9d842);
  status |= test__mulsf3(0x6f8f11bb, 0x4f97f48d, 0x7f800000);
  status |= test__mulsf3(0x208f11bb, 0x1f17f48d, 0x0054ec21);
  status |= test__mulsf3(0x3f8f11bb, 0x3f680b73, 0x3f81ae78);
  status |= test__mulsf3(0x6f8f11bb, 0x4f680b73, 0x7f800000);
  status |= test__mulsf3(0x208f11bb, 0x1ee80b73, 0x0040d73c);
  status |= test__mulsf3(0x3fff5dd8, 0x3f600000, 0x3fdf721d);
  status |= test__mulsf3(0x6fff5dd8, 0x4f600000, 0x7f800000);
  status |= test__mulsf3(0x20ff5dd8, 0x1ee00000, 0x006fb90e);
  status |= test__mulsf3(0x3fff5dd8, 0x3f100000, 0x3f8fa4ca);
  status |= test__mulsf3(0x6fff5dd8, 0x4f100000, 0x7f800000);
  status |= test__mulsf3(0x20ff5dd8, 0x1e900000, 0x0047d265);
  status |= test__mulsf3(0x3fffe96b, 0x3f7efb43, 0x3ffee4c5);
  status |= test__mulsf3(0x6fffe96b, 0x4f7efb43, 0x7f800000);
  status |= test__mulsf3(0x20ffe96b, 0x1efefb43, 0x007f7263);
  status |= test__mulsf3(0x3fffe96b, 0x3f0104bd, 0x3f80f95b);
  status |= test__mulsf3(0x6fffe96b, 0x4f0104bd, 0x7f800000);
  status |= test__mulsf3(0x20ffe96b, 0x1e8104bd, 0x00407cae);
  status |= test__mulsf3(0x3f8fbbb7, 0x3fa6edf9, 0x3fbb72aa);
  status |= test__mulsf3(0x6f8fbbb7, 0x4fa6edf9, 0x7f800000);
  status |= test__mulsf3(0x208fbbb7, 0x1f26edf9, 0x005db955);
  status |= test__mulsf3(0x3f8fbbb7, 0x3fd91207, 0x3ff3c07b);
  status |= test__mulsf3(0x6f8fbbb7, 0x4fd91207, 0x7f800000);
  status |= test__mulsf3(0x208fbbb7, 0x1f591207, 0x0079e03d);
  status |= test__mulsf3(0x3f8fbbb7, 0x3f991207, 0x3fabe29f);
  status |= test__mulsf3(0x6f8fbbb7, 0x4f991207, 0x7f800000);
  status |= test__mulsf3(0x208fbbb7, 0x1f191207, 0x0055f150);
  status |= test__mulsf3(0x3f8fbbb7, 0x3f66edf9, 0x3f81a843);
  status |= test__mulsf3(0x6f8fbbb7, 0x4f66edf9, 0x7f800000);
  status |= test__mulsf3(0x208fbbb7, 0x1ee6edf9, 0x0040d421);
  status |= test__mulsf3(0x3fdb62f3, 0x3f7879c5, 0x3fd4f036);
  status |= test__mulsf3(0x6fdb62f3, 0x4f7879c5, 0x7f800000);
  status |= test__mulsf3(0x20db62f3, 0x1ef879c5, 0x006a781b);
  status |= test__mulsf3(0x3faaea45, 0x3f8b6773, 0x3fba2489);
  status |= test__mulsf3(0x6faaea45, 0x4f8b6773, 0x7f800000);
  status |= test__mulsf3(0x20aaea45, 0x1f0b6773, 0x005d1244);
  status |= test__mulsf3(0x3fafa7ec, 0x3f900000, 0x3fc59cea);
  status |= test__mulsf3(0x6fafa7ec, 0x4f900000, 0x7f800000);
  status |= test__mulsf3(0x20afa7ec, 0x1f100000, 0x0062ce75);
  status |= test__mulsf3(0x3fcf8c8d, 0x3f271645, 0x3f8776be);
  status |= test__mulsf3(0x6fcf8c8d, 0x4f271645, 0x7f800000);
  status |= test__mulsf3(0x20cf8c8d, 0x1ea71645, 0x0043bb5f);
  status |= test__mulsf3(0x3fc173ef, 0x3f901b0f, 0x3fd9cb52);
  status |= test__mulsf3(0x6fc173ef, 0x4f901b0f, 0x7f800000);
  status |= test__mulsf3(0x20c173ef, 0x1f101b0f, 0x006ce5a9);
  status |= test__mulsf3(0x3fb48d33, 0x3f4a35fb, 0x3f8e9d7d);
  status |= test__mulsf3(0x6fb48d33, 0x4f4a35fb, 0x7f800000);
  status |= test__mulsf3(0x20b48d33, 0x1eca35fb, 0x00474ebe);
  status |= test__mulsf3(0x3fc6f87b, 0x3f65d94d, 0x3fb2a52a);
  status |= test__mulsf3(0x6fc6f87b, 0x4f65d94d, 0x7f800000);
  status |= test__mulsf3(0x20c6f87b, 0x1ee5d94d, 0x00595295);
  status |= test__mulsf3(0x3f860ae7, 0x3f969729, 0x3f9db312);
  status |= test__mulsf3(0x6f860ae7, 0x4f969729, 0x7f800000);
  status |= test__mulsf3(0x20860ae7, 0x1f169729, 0x004ed989);
  status |= test__mulsf3(0x3f860ae7, 0x3fc00000, 0x3fc9105a);
  status |= test__mulsf3(0x6f860ae7, 0x4fc00000, 0x7f800000);
  status |= test__mulsf3(0x20860ae7, 0x1f400000, 0x0064882d);
  status |= test__mulsf3(0x3f860ae7, 0x3fe968d7, 0x3ff46da3);
  status |= test__mulsf3(0x6f860ae7, 0x4fe968d7, 0x7f800000);
  status |= test__mulsf3(0x20860ae7, 0x1f6968d7, 0x007a36d1);
  status |= test__mulsf3(0x3f860ae7, 0x3f800000, 0x3f860ae7);
  status |= test__mulsf3(0x6f860ae7, 0x4f800000, 0x7f800000);
  status |= test__mulsf3(0x20860ae7, 0x1f000000, 0x00430574);
  status |= test__mulsf3(0x3f860ae7, 0x3fa968d7, 0x3fb1682f);
  status |= test__mulsf3(0x6f860ae7, 0x4fa968d7, 0x7f800000);
  status |= test__mulsf3(0x20860ae7, 0x1f2968d7, 0x0058b418);
  status |= test__mulsf3(0x3f860ae7, 0x3fd69729, 0x3fe0b886);
  status |= test__mulsf3(0x6f860ae7, 0x4fd69729, 0x7f800000);
  status |= test__mulsf3(0x20860ae7, 0x1f569729, 0x00705c43);
  status |= test__mulsf3(0x3f9aecdd, 0x3fb14b75, 0x3fd696de);
  status |= test__mulsf3(0x6f9aecdd, 0x4fb14b75, 0x7f800000);
  status |= test__mulsf3(0x209aecdd, 0x1f314b75, 0x006b4b6f);
  status |= test__mulsf3(0x3f9aecdd, 0x3fceb48b, 0x3ffa2fb9);
  status |= test__mulsf3(0x6f9aecdd, 0x4fceb48b, 0x7f800000);
  status |= test__mulsf3(0x209aecdd, 0x1f4eb48b, 0x007d17dc);
  status |= test__mulsf3(0x3f9aecdd, 0x3fc00000, 0x3fe8634c);
  status |= test__mulsf3(0x6f9aecdd, 0x4fc00000, 0x7f800000);
  status |= test__mulsf3(0x209aecdd, 0x1f400000, 0x007431a6);
  status |= test__mulsf3(0x3fd65dc6, 0x3f400000, 0x3fa0c654);
  status |= test__mulsf3(0x6fd65dc6, 0x4f400000, 0x7f800000);
  status |= test__mulsf3(0x20d65dc6, 0x1ec00000, 0x0050632a);
  status |= test__mulsf3(0x3feecf03, 0x3f5f93ab, 0x3fd09014);
  status |= test__mulsf3(0x6feecf03, 0x4f5f93ab, 0x7f800000);
  status |= test__mulsf3(0x20eecf03, 0x1edf93ab, 0x0068480a);
  status |= test__mulsf3(0x3feecf03, 0x3f206c55, 0x3f95a670);
  status |= test__mulsf3(0x6feecf03, 0x4f206c55, 0x7f800000);
  status |= test__mulsf3(0x20eecf03, 0x1ea06c55, 0x004ad338);
  status |= test__mulsf3(0x3f98feed, 0x3f60f11b, 0x3f866f27);
  status |= test__mulsf3(0x6f98feed, 0x4f60f11b, 0x7f800000);
  status |= test__mulsf3(0x2098feed, 0x1ee0f11b, 0x00433794);
  status |= test__mulsf3(0x3f9a1b9d, 0x3f9c42b5, 0x3fbc21f8);
  status |= test__mulsf3(0x6f9a1b9d, 0x4f9c42b5, 0x7f800000);
  status |= test__mulsf3(0x209a1b9d, 0x1f1c42b5, 0x005e10fc);
  status |= test__mulsf3(0x3f9a1b9d, 0x3f5c42b5, 0x3f8497e3);
  status |= test__mulsf3(0x6f9a1b9d, 0x4f5c42b5, 0x7f800000);
  status |= test__mulsf3(0x209a1b9d, 0x1edc42b5, 0x00424bf2);
  status |= test__mulsf3(0x3f947044, 0x3f600000, 0x3f81e23c);
  status |= test__mulsf3(0x6f947044, 0x4f600000, 0x7f800000);
  status |= test__mulsf3(0x20947044, 0x1ee00000, 0x0040f11e);
  status |= test__mulsf3(0x3fa3fb77, 0x3f6eb1b9, 0x3f98e5a0);
  status |= test__mulsf3(0x6fa3fb77, 0x4f6eb1b9, 0x7f800000);
  status |= test__mulsf3(0x20a3fb77, 0x1eeeb1b9, 0x004c72d0);
  status |= test__mulsf3(0x3fb291df, 0x3f466a1f, 0x3f8a66d9);
  status |= test__mulsf3(0x6fb291df, 0x4f466a1f, 0x7f800000);
  status |= test__mulsf3(0x20b291df, 0x1ec66a1f, 0x0045336c);
  status |= test__mulsf3(0x3fde13d5, 0x3f6b7283, 0x3fcc3f8b);
  status |= test__mulsf3(0x6fde13d5, 0x4f6b7283, 0x7f800000);
  status |= test__mulsf3(0x20de13d5, 0x1eeb7283, 0x00661fc5);
  status |= test__mulsf3(0x3fd5b211, 0x3f80810f, 0x3fd68987);
  status |= test__mulsf3(0x6fd5b211, 0x4f80810f, 0x7f800000);
  status |= test__mulsf3(0x20d5b211, 0x1f00810f, 0x006b44c4);
  status |= test__mulsf3(0x3fd5b211, 0x3f3f7ef1, 0x3f9fd9d2);
  status |= test__mulsf3(0x6fd5b211, 0x4f3f7ef1, 0x7f800000);
  status |= test__mulsf3(0x20d5b211, 0x1ebf7ef1, 0x004fece9);
  status |= test__mulsf3(0x3fadfbc4, 0x3f400000, 0x3f827cd3);
  status |= test__mulsf3(0x6fadfbc4, 0x4f400000, 0x7f800000);
  status |= test__mulsf3(0x20adfbc4, 0x1ec00000, 0x00413e6a);
  status |= test__mulsf3(0x3fd0ef03, 0x3f800000, 0x3fd0ef03);
  status |= test__mulsf3(0x6fd0ef03, 0x4f800000, 0x7f800000);
  status |= test__mulsf3(0x20d0ef03, 0x1f000000, 0x00687782);
  status |= test__mulsf3(0x3fd0ef03, 0x3f8673ab, 0x3fdb7705);
  status |= test__mulsf3(0x6fd0ef03, 0x4f8673ab, 0x7f800000);
  status |= test__mulsf3(0x20d0ef03, 0x1f0673ab, 0x006dbb83);
  status |= test__mulsf3(0x3fd0ef03, 0x3f798c55, 0x3fcbab02);
  status |= test__mulsf3(0x6fd0ef03, 0x4f798c55, 0x7f800000);
  status |= test__mulsf3(0x20d0ef03, 0x1ef98c55, 0x0065d581);
  status |= test__mulsf3(0x3fdd1181, 0x3f8ad17f, 0x3fefc0b1);
  status |= test__mulsf3(0x6fdd1181, 0x4f8ad17f, 0x7f800000);
  status |= test__mulsf3(0x20dd1181, 0x1f0ad17f, 0x0077e058);
  status |= test__mulsf3(0x3fdd1181, 0x3f752e81, 0x3fd3b9e9);
  status |= test__mulsf3(0x6fdd1181, 0x4f752e81, 0x7f800000);
  status |= test__mulsf3(0x20dd1181, 0x1ef52e81, 0x0069dcf5);
  status |= test__mulsf3(0x3f92efc6, 0x3fa00000, 0x3fb7abb8);
  status |= test__mulsf3(0x6f92efc6, 0x4fa00000, 0x7f800000);
  status |= test__mulsf3(0x2092efc6, 0x1f200000, 0x005bd5dc);
  status |= test__mulsf3(0x3fdcefe6, 0x3f400000, 0x3fa5b3ec);
  status |= test__mulsf3(0x6fdcefe6, 0x4f400000, 0x7f800000);
  status |= test__mulsf3(0x20dcefe6, 0x1ec00000, 0x0052d9f6);
  status |= test__mulsf3(0x3fad6507, 0x3fa2f8b7, 0x3fdcc4c9);
  status |= test__mulsf3(0x6fad6507, 0x4fa2f8b7, 0x7f800000);
  status |= test__mulsf3(0x20ad6507, 0x1f22f8b7, 0x006e6264);
  status |= test__mulsf3(0x3fad6507, 0x3f62f8b7, 0x3f99bba6);
  status |= test__mulsf3(0x6fad6507, 0x4f62f8b7, 0x7f800000);
  status |= test__mulsf3(0x20ad6507, 0x1ee2f8b7, 0x004cddd3);
  status |= test__mulsf3(0x3fbfde6b, 0x3f8721bd, 0x3fca8f27);
  status |= test__mulsf3(0x6fbfde6b, 0x4f8721bd, 0x7f800000);
  status |= test__mulsf3(0x20bfde6b, 0x1f0721bd, 0x00654794);
  status |= test__mulsf3(0x3fbfde6b, 0x3f4721bd, 0x3f953f2e);
  status |= test__mulsf3(0x6fbfde6b, 0x4f4721bd, 0x7f800000);
  status |= test__mulsf3(0x20bfde6b, 0x1ec721bd, 0x004a9f97);
  status |= test__mulsf3(0x3ff40db4, 0x3f400000, 0x3fb70a47);
  status |= test__mulsf3(0x6ff40db4, 0x4f400000, 0x7f800000);
  status |= test__mulsf3(0x20f40db4, 0x1ec00000, 0x005b8524);
  status |= test__mulsf3(0x3ff40db4, 0x3f600000, 0x3fd58bfe);
  status |= test__mulsf3(0x6ff40db4, 0x4f600000, 0x7f800000);
  status |= test__mulsf3(0x20f40db4, 0x1ee00000, 0x006ac5ff);
  status |= test__mulsf3(0x3f9e20d3, 0x3f90c8a5, 0x3fb2dccc);
  status |= test__mulsf3(0x6f9e20d3, 0x4f90c8a5, 0x7f800000);
  status |= test__mulsf3(0x209e20d3, 0x1f10c8a5, 0x00596e66);
  status |= test__mulsf3(0x3f9e20d3, 0x3fc00000, 0x3fed313c);
  status |= test__mulsf3(0x6f9e20d3, 0x4fc00000, 0x7f800000);
  status |= test__mulsf3(0x209e20d3, 0x1f400000, 0x0076989e);
  status |= test__mulsf3(0x3f9e20d3, 0x3f50c8a5, 0x3f80f69b);
  status |= test__mulsf3(0x6f9e20d3, 0x4f50c8a5, 0x7f800000);
  status |= test__mulsf3(0x209e20d3, 0x1ed0c8a5, 0x00407b4d);
  status |= test__mulsf3(0x3f82e641, 0x3f8fd63f, 0x3f931856);
  status |= test__mulsf3(0x6f82e641, 0x4f8fd63f, 0x7f800000);
  status |= test__mulsf3(0x2082e641, 0x1f0fd63f, 0x00498c2b);
  status |= test__mulsf3(0x3f9a1901, 0x3f96e701, 0x3fb5ab68);
  status |= test__mulsf3(0x6f9a1901, 0x4f96e701, 0x7f800000);
  status |= test__mulsf3(0x209a1901, 0x1f16e701, 0x005ad5b4);
  status |= test__mulsf3(0x3fa21aa1, 0x3f7c4961, 0x3f9fc0ae);
  status |= test__mulsf3(0x6fa21aa1, 0x4f7c4961, 0x7f800000);
  status |= test__mulsf3(0x20a21aa1, 0x1efc4961, 0x004fe057);
  status |= test__mulsf3(0x3fcd0767, 0x3f782457, 0x3fc6bc47);
  status |= test__mulsf3(0x6fcd0767, 0x4f782457, 0x7f800000);
  status |= test__mulsf3(0x20cd0767, 0x1ef82457, 0x00635e23);
  status |= test__mulsf3(0x3fb875e1, 0x3f968e21, 0x3fd8f6f6);
  status |= test__mulsf3(0x6fb875e1, 0x4f968e21, 0x7f800000);
  status |= test__mulsf3(0x20b875e1, 0x1f168e21, 0x006c7b7b);
  status |= test__mulsf3(0x3fc2f0d7, 0x3f5efd19, 0x3fa9cd95);
  status |= test__mulsf3(0x6fc2f0d7, 0x4f5efd19, 0x7f800000);
  status |= test__mulsf3(0x20c2f0d7, 0x1edefd19, 0x0054e6cb);
  status |= test__mulsf3(0x7f7ffffe, 0x3f800001, 0x7f800000);
  status |= test__mulsf3(0x00000003, 0xc00fffff, 0x80000007);
  status |= test__mulsf3(0x00000003, 0x400fffff, 0x00000007);
  status |= test__mulsf3(0x80000003, 0xc00fffff, 0x00000007);
  status |= test__mulsf3(0x80000003, 0x400fffff, 0x80000007);
  status |= test__mulsf3(0x00000003, 0xc00ffffd, 0x80000007);
  status |= test__mulsf3(0x00000003, 0x400ffffd, 0x00000007);
  status |= test__mulsf3(0x80000003, 0xc00ffffd, 0x00000007);
  status |= test__mulsf3(0x80000003, 0x400ffffd, 0x80000007);
  status |= test__mulsf3(0x3e00007f, 0x017c0000, 0x003f003f);
  status |= test__mulsf3(0xcf7fff00, 0xc0ffff00, 0x50fffe00);
  status |= test__mulsf3(0x3fdf7f00, 0x3fffff00, 0x405f7e21);
  status |= test__mulsf3(0x19b92144, 0x1a310000, 0x00000001);
  status |= test__mulsf3(0x19ffc008, 0x1a002004, 0x00000001);
  status |= test__mulsf3(0x7f7ffff0, 0xc0000008, 0xff800000);

  // Test that the result of an operation is a NaN at all when it should be.
  //
  // In most configurations these tests' results are checked compared using
  // compareResultF, so we set all the answers to the canonical NaN 0x7fc00000,
  // which causes compareResultF to accept any NaN encoding. We also use the
  // same value as the input NaN in tests that have one, so that even in
  // EXPECT_EXACT_RESULTS mode these tests should pass, because 0x7fc00000 is
  // still the exact expected NaN.
  status |= test__mulsf3(0x7f800000, 0x00000000, 0x7fc00000);
  status |= test__mulsf3(0x7f800000, 0x80000000, 0x7fc00000);
  status |= test__mulsf3(0x80000000, 0x7f800000, 0x7fc00000);
  status |= test__mulsf3(0x80000000, 0xff800000, 0x7fc00000);
  status |= test__mulsf3(0x3f800000, 0x7fc00000, 0x7fc00000);
  status |= test__mulsf3(0x7fc00000, 0x3f800000, 0x7fc00000);
  status |= test__mulsf3(0x7fc00000, 0x7fc00000, 0x7fc00000);

#ifdef ARM_NAN_HANDLING
  // Tests specific to the NaN handling of Arm hardware, mimicked by
  // arm/mulsf3.S:
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
  //    NaN with fewest bits set, 0x7fc00000.

  status |= test__mulsf3(0x00000000, 0x7fad4be3, 0x7fed4be3);
  status |= test__mulsf3(0x00000000, 0x7fdf48c7, 0x7fdf48c7);
  status |= test__mulsf3(0x00000001, 0x7f970eba, 0x7fd70eba);
  status |= test__mulsf3(0x00000001, 0x7fc35716, 0x7fc35716);
  status |= test__mulsf3(0x007fffff, 0x7fbf52d6, 0x7fff52d6);
  status |= test__mulsf3(0x007fffff, 0x7fc7a2df, 0x7fc7a2df);
  status |= test__mulsf3(0x3f800000, 0x7f987a85, 0x7fd87a85);
  status |= test__mulsf3(0x3f800000, 0x7fc50124, 0x7fc50124);
  status |= test__mulsf3(0x7f7fffff, 0x7f95fd6f, 0x7fd5fd6f);
  status |= test__mulsf3(0x7f7fffff, 0x7ffc28dc, 0x7ffc28dc);
  status |= test__mulsf3(0x7f800000, 0x00000000, 0x7fc00000);
  status |= test__mulsf3(0x7f800000, 0x7f8dd790, 0x7fcdd790);
  status |= test__mulsf3(0x7f800000, 0x7fd2ef2b, 0x7fd2ef2b);
  status |= test__mulsf3(0x7f800000, 0x80000000, 0x7fc00000);
  status |= test__mulsf3(0x7f99b09d, 0x00000000, 0x7fd9b09d);
  status |= test__mulsf3(0x7f93541e, 0x00000001, 0x7fd3541e);
  status |= test__mulsf3(0x7f9fc002, 0x007fffff, 0x7fdfc002);
  status |= test__mulsf3(0x7fb5db77, 0x3f800000, 0x7ff5db77);
  status |= test__mulsf3(0x7f9f5d92, 0x7f7fffff, 0x7fdf5d92);
  status |= test__mulsf3(0x7fac7a36, 0x7f800000, 0x7fec7a36);
  status |= test__mulsf3(0x7fb42008, 0x7fb0ee07, 0x7ff42008);
  status |= test__mulsf3(0x7f8bd740, 0x7fc7aaf1, 0x7fcbd740);
  status |= test__mulsf3(0x7f9bb57b, 0x80000000, 0x7fdbb57b);
  status |= test__mulsf3(0x7f951a78, 0x80000001, 0x7fd51a78);
  status |= test__mulsf3(0x7f9ba63b, 0x807fffff, 0x7fdba63b);
  status |= test__mulsf3(0x7f89463c, 0xbf800000, 0x7fc9463c);
  status |= test__mulsf3(0x7fb63563, 0xff7fffff, 0x7ff63563);
  status |= test__mulsf3(0x7f90886e, 0xff800000, 0x7fd0886e);
  status |= test__mulsf3(0x7fe8c15e, 0x00000000, 0x7fe8c15e);
  status |= test__mulsf3(0x7fe915ae, 0x00000001, 0x7fe915ae);
  status |= test__mulsf3(0x7ffa9b42, 0x007fffff, 0x7ffa9b42);
  status |= test__mulsf3(0x7fdad0f5, 0x3f800000, 0x7fdad0f5);
  status |= test__mulsf3(0x7fd10dcb, 0x7f7fffff, 0x7fd10dcb);
  status |= test__mulsf3(0x7fd08e8a, 0x7f800000, 0x7fd08e8a);
  status |= test__mulsf3(0x7fc3a9e6, 0x7f91a816, 0x7fd1a816);
  status |= test__mulsf3(0x7fdb229c, 0x7fc26c68, 0x7fdb229c);
  status |= test__mulsf3(0x7fc9f6bb, 0x80000000, 0x7fc9f6bb);
  status |= test__mulsf3(0x7ffa178b, 0x80000001, 0x7ffa178b);
  status |= test__mulsf3(0x7fef2a0b, 0x807fffff, 0x7fef2a0b);
  status |= test__mulsf3(0x7ffc885b, 0xbf800000, 0x7ffc885b);
  status |= test__mulsf3(0x7fd26e8c, 0xff7fffff, 0x7fd26e8c);
  status |= test__mulsf3(0x7fc55329, 0xff800000, 0x7fc55329);
  status |= test__mulsf3(0x80000000, 0x7f800000, 0x7fc00000);
  status |= test__mulsf3(0x80000000, 0x7fa833ae, 0x7fe833ae);
  status |= test__mulsf3(0x80000000, 0x7fc4df63, 0x7fc4df63);
  status |= test__mulsf3(0x80000000, 0xff800000, 0x7fc00000);
  status |= test__mulsf3(0x80000001, 0x7f98827d, 0x7fd8827d);
  status |= test__mulsf3(0x80000001, 0x7fd7acc5, 0x7fd7acc5);
  status |= test__mulsf3(0x807fffff, 0x7fad19c0, 0x7fed19c0);
  status |= test__mulsf3(0x807fffff, 0x7ffe1907, 0x7ffe1907);
  status |= test__mulsf3(0xbf800000, 0x7fa95487, 0x7fe95487);
  status |= test__mulsf3(0xbf800000, 0x7fd2bbee, 0x7fd2bbee);
  status |= test__mulsf3(0xff7fffff, 0x7f86ba21, 0x7fc6ba21);
  status |= test__mulsf3(0xff7fffff, 0x7feb00d7, 0x7feb00d7);
  status |= test__mulsf3(0xff800000, 0x7f857fdc, 0x7fc57fdc);
  status |= test__mulsf3(0xff800000, 0x7fde0397, 0x7fde0397);
#endif // ARM_NAN_HANDLING

  return status;
}
