// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_addsf3

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

// By default this test uses compareResultF to check the returned floats, which
// accepts any returned NaN if the expected result is the canonical NaN value
// 0x7fc00000. For the Thumb1 optimized FP implementation, which commits to a
// more detailed handling of NaNs, we tighten up the check and include some
// extra test cases specific to that NaN policy.
#if __thumb__ && !__thumb2__
#  define EXPECT_EXACT_RESULTS
#  define ARM_NAN_HANDLING
#endif

// Returns: a - b
COMPILER_RT_ABI float __subsf3(float a, float b);

int test__subsf3(uint32_t a_rep, uint32_t b_rep, uint32_t expected_rep) {
  float a = fromRep32(a_rep), b = fromRep32(b_rep);
  float x = __subsf3(a, b);
#ifdef EXPECT_EXACT_RESULTS
  int ret = toRep32(x) == expected_rep;
#else
  int ret = compareResultF(x, expected_rep);
#endif

  if (ret) {
    printf("error in test__subsf3(%08" PRIx32 ", %08" PRIx32 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           a_rep, b_rep, toRep32(x), expected_rep);
  }
  return ret;
}

int main() {
  int status = 0;

  status |= test__subsf3(0x00000000, 0x00000000, 0x00000000);
  status |= test__subsf3(0x00000000, 0x007fffff, 0x807fffff);
  status |= test__subsf3(0x00000000, 0x00800000, 0x80800000);
  status |= test__subsf3(0x00000000, 0x7f800000, 0xff800000);
  status |= test__subsf3(0x00000000, 0x80000000, 0x00000000);
  status |= test__subsf3(0x00000000, 0x807fffff, 0x007fffff);
  status |= test__subsf3(0x00000000, 0xbf800000, 0x3f800000);
  status |= test__subsf3(0x00000000, 0xff000000, 0x7f000000);
  status |= test__subsf3(0x00000000, 0xff800000, 0x7f800000);
  status |= test__subsf3(0x00000001, 0x00000001, 0x00000000);
  status |= test__subsf3(0x00000001, 0x80000001, 0x00000002);
  status |= test__subsf3(0x00000001, 0xbf7fffff, 0x3f7fffff);
  status |= test__subsf3(0x00000001, 0xbf800000, 0x3f800000);
  status |= test__subsf3(0x00000001, 0xbffffffe, 0x3ffffffe);
  status |= test__subsf3(0x00000001, 0xbfffffff, 0x3fffffff);
  status |= test__subsf3(0x00000001, 0xfeffffff, 0x7effffff);
  status |= test__subsf3(0x00000001, 0xff000000, 0x7f000000);
  status |= test__subsf3(0x00000001, 0xff7ffffe, 0x7f7ffffe);
  status |= test__subsf3(0x00000001, 0xff7fffff, 0x7f7fffff);
  status |= test__subsf3(0x00000002, 0x00000001, 0x00000001);
  status |= test__subsf3(0x00000003, 0x00000000, 0x00000003);
  status |= test__subsf3(0x00000003, 0x00000002, 0x00000001);
  status |= test__subsf3(0x00000003, 0x40a00000, 0xc0a00000);
  status |= test__subsf3(0x00000003, 0x7f000000, 0xff000000);
  status |= test__subsf3(0x00000003, 0x7f800000, 0xff800000);
  status |= test__subsf3(0x00000003, 0x80000000, 0x00000003);
  status |= test__subsf3(0x00000003, 0xff800000, 0x7f800000);
  status |= test__subsf3(0x00000004, 0x80000004, 0x00000008);
  status |= test__subsf3(0x007ffffc, 0x007ffffc, 0x00000000);
  status |= test__subsf3(0x007ffffd, 0x007ffffe, 0x80000001);
  status |= test__subsf3(0x007fffff, 0x007ffffe, 0x00000001);
  status |= test__subsf3(0x007fffff, 0x00800000, 0x80000001);
  status |= test__subsf3(0x007fffff, 0x807fffff, 0x00fffffe);
  status |= test__subsf3(0x00800000, 0x00800000, 0x00000000);
  status |= test__subsf3(0x00800000, 0x80000000, 0x00800000);
  status |= test__subsf3(0x00800000, 0x80800000, 0x01000000);
  status |= test__subsf3(0x00800001, 0x00800000, 0x00000001);
  status |= test__subsf3(0x00800001, 0x00800002, 0x80000001);
  status |= test__subsf3(0x00ffffff, 0x01000000, 0x80000001);
  status |= test__subsf3(0x00ffffff, 0x01000002, 0x80000005);
  status |= test__subsf3(0x00ffffff, 0x01000004, 0x80000009);
  status |= test__subsf3(0x01000000, 0x00ffffff, 0x00000001);
  status |= test__subsf3(0x01000001, 0x00800001, 0x00800001);
  status |= test__subsf3(0x01000001, 0x00ffffff, 0x00000003);
  status |= test__subsf3(0x01000002, 0x00800001, 0x00800003);
  status |= test__subsf3(0x017fffff, 0x01800000, 0x80000002);
  status |= test__subsf3(0x01800000, 0x017fffff, 0x00000002);
  status |= test__subsf3(0x01800001, 0x017fffff, 0x00000006);
  status |= test__subsf3(0x01800002, 0x01000003, 0x01000001);
  status |= test__subsf3(0x3f7fffff, 0x00000001, 0x3f7fffff);
  status |= test__subsf3(0x3f800000, 0x00000000, 0x3f800000);
  status |= test__subsf3(0x3f800000, 0x3f800000, 0x00000000);
  status |= test__subsf3(0x3f800000, 0xbf800000, 0x40000000);
  status |= test__subsf3(0x3f800000, 0xbf800003, 0x40000002);
  status |= test__subsf3(0x3f800000, 0xc0000000, 0x40400000);
  status |= test__subsf3(0x3f800000, 0xc0e00000, 0x41000000);
  status |= test__subsf3(0x3f800001, 0x3f800000, 0x34000000);
  status |= test__subsf3(0x3f800001, 0x3f800002, 0xb4000000);
  status |= test__subsf3(0x3f800001, 0xbf800000, 0x40000000);
  status |= test__subsf3(0x3ffffffc, 0x3ffffffd, 0xb4000000);
  status |= test__subsf3(0x3fffffff, 0x40000000, 0xb4000000);
  status |= test__subsf3(0x40000000, 0x3fffffff, 0x34000000);
  status |= test__subsf3(0x40000000, 0x40000000, 0x00000000);
  status |= test__subsf3(0x40000000, 0x40000001, 0xb4800000);
  status |= test__subsf3(0x40000000, 0x40a00000, 0xc0400000);
  status |= test__subsf3(0x40000000, 0xb4000000, 0x40000000);
  status |= test__subsf3(0x40000000, 0xbf800000, 0x40400000);
  status |= test__subsf3(0x40000000, 0xc0000000, 0x40800000);
  status |= test__subsf3(0x40000000, 0xc0000001, 0x40800000);
  status |= test__subsf3(0x40000001, 0x3f800001, 0x3f800001);
  status |= test__subsf3(0x40000001, 0xb4000000, 0x40000002);
  status |= test__subsf3(0x40000001, 0xc0000002, 0x40800002);
  status |= test__subsf3(0x40000002, 0x3f800001, 0x3f800003);
  status |= test__subsf3(0x40000002, 0x3f800003, 0x3f800001);
  status |= test__subsf3(0x40000004, 0x40000003, 0x34800000);
  status |= test__subsf3(0x40400000, 0xc0400000, 0x40c00000);
  status |= test__subsf3(0x407fffff, 0x407ffffe, 0x34800000);
  status |= test__subsf3(0x407fffff, 0x40800002, 0xb5a00000);
  status |= test__subsf3(0x407fffff, 0xb3ffffff, 0x407fffff);
  status |= test__subsf3(0x407fffff, 0xb4000000, 0x40800000);
  status |= test__subsf3(0x40800001, 0x407fffff, 0x35400000);
  status |= test__subsf3(0x40a00000, 0x00000000, 0x40a00000);
  status |= test__subsf3(0x40a00000, 0x3f800000, 0x40800000);
  status |= test__subsf3(0x40a00000, 0x40a00000, 0x00000000);
  status |= test__subsf3(0x40a00000, 0x80000000, 0x40a00000);
  status |= test__subsf3(0x7d800001, 0x7d7fffff, 0x72400000);
  status |= test__subsf3(0x7e7fffff, 0x7e7ffffe, 0x72800000);
  status |= test__subsf3(0x7e7fffff, 0x7e800002, 0xf3a00000);
  status |= test__subsf3(0x7e800000, 0x7e7fffff, 0x72800000);
  status |= test__subsf3(0x7e800000, 0x7e800001, 0xf3000000);
  status |= test__subsf3(0x7e800000, 0xfe800000, 0x7f000000);
  status |= test__subsf3(0x7e800001, 0x7f000001, 0xfe800001);
  status |= test__subsf3(0x7e800001, 0xfe800000, 0x7f000000);
  status |= test__subsf3(0x7e800002, 0x7e000003, 0x7e000001);
  status |= test__subsf3(0x7e800004, 0x7e800003, 0x73000000);
  status |= test__subsf3(0x7efffffe, 0xfefffffe, 0x7f7ffffe);
  status |= test__subsf3(0x7efffffe, 0xfeffffff, 0x7f7ffffe);
  status |= test__subsf3(0x7effffff, 0x3f800000, 0x7effffff);
  status |= test__subsf3(0x7effffff, 0x7f000000, 0xf3000000);
  status |= test__subsf3(0x7effffff, 0xbf800000, 0x7effffff);
  status |= test__subsf3(0x7effffff, 0xff000000, 0x7f800000);
  status |= test__subsf3(0x7f000000, 0x3f800000, 0x7f000000);
  status |= test__subsf3(0x7f000000, 0x7f000000, 0x00000000);
  status |= test__subsf3(0x7f000000, 0x7f800000, 0xff800000);
  status |= test__subsf3(0x7f000000, 0xbf800000, 0x7f000000);
  status |= test__subsf3(0x7f000000, 0xff000000, 0x7f800000);
  status |= test__subsf3(0x7f000000, 0xff800000, 0x7f800000);
  status |= test__subsf3(0x7f000001, 0x7f000000, 0x73800000);
  status |= test__subsf3(0x7f000001, 0x7f000002, 0xf3800000);
  status |= test__subsf3(0x7f000001, 0xff000000, 0x7f800000);
  status |= test__subsf3(0x7f000002, 0x7e800001, 0x7e800003);
  status |= test__subsf3(0x7f7ffffe, 0x3f800000, 0x7f7ffffe);
  status |= test__subsf3(0x7f7ffffe, 0x7f7fffff, 0xf3800000);
  status |= test__subsf3(0x7f7ffffe, 0xbf800000, 0x7f7ffffe);
  status |= test__subsf3(0x7f7ffffe, 0xff7ffffe, 0x7f800000);
  status |= test__subsf3(0x7f7ffffe, 0xff7fffff, 0x7f800000);
  status |= test__subsf3(0x7f7fffff, 0x00000001, 0x7f7fffff);
  status |= test__subsf3(0x7f7fffff, 0x3f800000, 0x7f7fffff);
  status |= test__subsf3(0x7f7fffff, 0x7f7fffff, 0x00000000);
  status |= test__subsf3(0x7f7fffff, 0xbf800000, 0x7f7fffff);
  status |= test__subsf3(0x7f800000, 0x00000000, 0x7f800000);
  status |= test__subsf3(0x7f800000, 0x007fffff, 0x7f800000);
  status |= test__subsf3(0x7f800000, 0x7f000000, 0x7f800000);
  status |= test__subsf3(0x7f800000, 0x80000000, 0x7f800000);
  status |= test__subsf3(0x7f800000, 0x807fffff, 0x7f800000);
  status |= test__subsf3(0x7f800000, 0xff000000, 0x7f800000);
  status |= test__subsf3(0x7f800000, 0xff800000, 0x7f800000);
  status |= test__subsf3(0x80000000, 0x00000000, 0x80000000);
  status |= test__subsf3(0x80000000, 0x007fffff, 0x807fffff);
  status |= test__subsf3(0x80000000, 0x00800000, 0x80800000);
  status |= test__subsf3(0x80000000, 0x3f800000, 0xbf800000);
  status |= test__subsf3(0x80000000, 0x7f800000, 0xff800000);
  status |= test__subsf3(0x80000000, 0x80000000, 0x00000000);
  status |= test__subsf3(0x80000000, 0x807fffff, 0x007fffff);
  status |= test__subsf3(0x80000000, 0xff000000, 0x7f000000);
  status |= test__subsf3(0x80000000, 0xff800000, 0x7f800000);
  status |= test__subsf3(0x80000001, 0x00000001, 0x80000002);
  status |= test__subsf3(0x80000001, 0x3f7fffff, 0xbf7fffff);
  status |= test__subsf3(0x80000001, 0x3f800000, 0xbf800000);
  status |= test__subsf3(0x80000001, 0x3ffffffe, 0xbffffffe);
  status |= test__subsf3(0x80000001, 0x3fffffff, 0xbfffffff);
  status |= test__subsf3(0x80000001, 0x7effffff, 0xfeffffff);
  status |= test__subsf3(0x80000001, 0x7f000000, 0xff000000);
  status |= test__subsf3(0x80000001, 0x7f7ffffe, 0xff7ffffe);
  status |= test__subsf3(0x80000001, 0x7f7fffff, 0xff7fffff);
  status |= test__subsf3(0x80000001, 0x80000001, 0x00000000);
  status |= test__subsf3(0x80000002, 0x80000001, 0x80000001);
  status |= test__subsf3(0x80000003, 0x00000000, 0x80000003);
  status |= test__subsf3(0x80000003, 0x7f800000, 0xff800000);
  status |= test__subsf3(0x80000003, 0x80000000, 0x80000003);
  status |= test__subsf3(0x80000003, 0x80000002, 0x80000001);
  status |= test__subsf3(0x80000003, 0xc0400000, 0x40400000);
  status |= test__subsf3(0x80000003, 0xff000000, 0x7f000000);
  status |= test__subsf3(0x80000003, 0xff800000, 0x7f800000);
  status |= test__subsf3(0x80000004, 0x00000004, 0x80000008);
  status |= test__subsf3(0x807ffffd, 0x807ffffe, 0x00000001);
  status |= test__subsf3(0x807fffff, 0x007fffff, 0x80fffffe);
  status |= test__subsf3(0x807fffff, 0x807ffffe, 0x80000001);
  status |= test__subsf3(0x807fffff, 0x807fffff, 0x00000000);
  status |= test__subsf3(0x807fffff, 0x80800000, 0x00000001);
  status |= test__subsf3(0x80800000, 0x80000000, 0x80800000);
  status |= test__subsf3(0x80800000, 0x80800000, 0x00000000);
  status |= test__subsf3(0x80800001, 0x80800000, 0x80000001);
  status |= test__subsf3(0x80800001, 0x80800002, 0x00000001);
  status |= test__subsf3(0x80ffffff, 0x81000000, 0x00000001);
  status |= test__subsf3(0x80ffffff, 0x81000002, 0x00000005);
  status |= test__subsf3(0x80ffffff, 0x81000004, 0x00000009);
  status |= test__subsf3(0x81000000, 0x80ffffff, 0x80000001);
  status |= test__subsf3(0x81000001, 0x80800001, 0x80800001);
  status |= test__subsf3(0x81000001, 0x80ffffff, 0x80000003);
  status |= test__subsf3(0x81000002, 0x80800001, 0x80800003);
  status |= test__subsf3(0x817fffff, 0x81800000, 0x00000002);
  status |= test__subsf3(0x81800000, 0x817fffff, 0x80000002);
  status |= test__subsf3(0x81800001, 0x817fffff, 0x80000006);
  status |= test__subsf3(0x81800002, 0x81000003, 0x81000001);
  status |= test__subsf3(0xbf800000, 0x00000000, 0xbf800000);
  status |= test__subsf3(0xbf800000, 0x3f800003, 0xc0000002);
  status |= test__subsf3(0xbf800001, 0x3f800000, 0xc0000000);
  status |= test__subsf3(0xbf800001, 0xbf800000, 0xb4000000);
  status |= test__subsf3(0xbf800001, 0xbf800002, 0x34000000);
  status |= test__subsf3(0xbffffffc, 0xbffffffd, 0x34000000);
  status |= test__subsf3(0xbfffffff, 0x80000001, 0xbfffffff);
  status |= test__subsf3(0xbfffffff, 0xc0000000, 0x34000000);
  status |= test__subsf3(0xc0000000, 0x40000001, 0xc0800000);
  status |= test__subsf3(0xc0000000, 0xbfffffff, 0xb4000000);
  status |= test__subsf3(0xc0000000, 0xc0000001, 0x34800000);
  status |= test__subsf3(0xc0000001, 0x40000002, 0xc0800002);
  status |= test__subsf3(0xc0000001, 0xbf800001, 0xbf800001);
  status |= test__subsf3(0xc0000002, 0xbf800001, 0xbf800003);
  status |= test__subsf3(0xc0000002, 0xbf800003, 0xbf800001);
  status |= test__subsf3(0xc0000004, 0xc0000003, 0xb4800000);
  status |= test__subsf3(0xc0400000, 0xc0400000, 0x00000000);
  status |= test__subsf3(0xc07fffff, 0x33ffffff, 0xc07fffff);
  status |= test__subsf3(0xc07fffff, 0x34000000, 0xc0800000);
  status |= test__subsf3(0xc07fffff, 0xc07ffffe, 0xb4800000);
  status |= test__subsf3(0xc07fffff, 0xc0800002, 0x35a00000);
  status |= test__subsf3(0xc0800001, 0xc07fffff, 0xb5400000);
  status |= test__subsf3(0xfd800001, 0xfd7fffff, 0xf2400000);
  status |= test__subsf3(0xfe7fffff, 0xfe7ffffe, 0xf2800000);
  status |= test__subsf3(0xfe7fffff, 0xfe800002, 0x73a00000);
  status |= test__subsf3(0xfe800000, 0xfe7fffff, 0xf2800000);
  status |= test__subsf3(0xfe800000, 0xfe800001, 0x73000000);
  status |= test__subsf3(0xfe800001, 0x7e800000, 0xff000000);
  status |= test__subsf3(0xfe800001, 0xff000001, 0x7e800001);
  status |= test__subsf3(0xfe800002, 0xfe000003, 0xfe000001);
  status |= test__subsf3(0xfe800004, 0xfe800003, 0xf3000000);
  status |= test__subsf3(0xfefffffe, 0x7efffffe, 0xff7ffffe);
  status |= test__subsf3(0xfefffffe, 0x7effffff, 0xff7ffffe);
  status |= test__subsf3(0xfefffffe, 0xfefffffe, 0x00000000);
  status |= test__subsf3(0xfeffffff, 0x3f800000, 0xfeffffff);
  status |= test__subsf3(0xfeffffff, 0x7f000000, 0xff800000);
  status |= test__subsf3(0xfeffffff, 0xbf800000, 0xfeffffff);
  status |= test__subsf3(0xfeffffff, 0xff000000, 0x73000000);
  status |= test__subsf3(0xff000000, 0x00000000, 0xff000000);
  status |= test__subsf3(0xff000000, 0x3f800000, 0xff000000);
  status |= test__subsf3(0xff000000, 0x7f000000, 0xff800000);
  status |= test__subsf3(0xff000000, 0x7f800000, 0xff800000);
  status |= test__subsf3(0xff000000, 0x80000000, 0xff000000);
  status |= test__subsf3(0xff000000, 0xbf800000, 0xff000000);
  status |= test__subsf3(0xff000000, 0xff800000, 0x7f800000);
  status |= test__subsf3(0xff000001, 0x7f000000, 0xff800000);
  status |= test__subsf3(0xff000001, 0xff000000, 0xf3800000);
  status |= test__subsf3(0xff000001, 0xff000002, 0x73800000);
  status |= test__subsf3(0xff000002, 0xfe800001, 0xfe800003);
  status |= test__subsf3(0xff7ffffe, 0x3f800000, 0xff7ffffe);
  status |= test__subsf3(0xff7ffffe, 0x7f7ffffe, 0xff800000);
  status |= test__subsf3(0xff7ffffe, 0x7f7fffff, 0xff800000);
  status |= test__subsf3(0xff7ffffe, 0xbf800000, 0xff7ffffe);
  status |= test__subsf3(0xff7ffffe, 0xff7fffff, 0x73800000);
  status |= test__subsf3(0xff7fffff, 0x3f800000, 0xff7fffff);
  status |= test__subsf3(0xff7fffff, 0x80000001, 0xff7fffff);
  status |= test__subsf3(0xff7fffff, 0xbf800000, 0xff7fffff);
  status |= test__subsf3(0xff800000, 0x00000000, 0xff800000);
  status |= test__subsf3(0xff800000, 0x007fffff, 0xff800000);
  status |= test__subsf3(0xff800000, 0x7f000000, 0xff800000);
  status |= test__subsf3(0xff800000, 0x7f800000, 0xff800000);
  status |= test__subsf3(0xff800000, 0x80000000, 0xff800000);
  status |= test__subsf3(0xff800000, 0x807fffff, 0xff800000);
  status |= test__subsf3(0xff800000, 0xff000000, 0xff800000);
  status |= test__subsf3(0x46f99cee, 0x4656466d, 0x468e79b8);
  status |= test__subsf3(0x007ffff7, 0x00f7ffff, 0x80780008);
  status |= test__subsf3(0x80ffffbf, 0x80800000, 0x807fffbf);

  // Test that the result of an operation is a NaN at all when it should be.
  //
  // In most configurations these tests' results are checked compared using
  // compareResultF, so we set all the answers to the canonical NaN 0x7fc00000,
  // which causes compareResultF to accept any NaN encoding. We also use the
  // same value as the input NaN in tests that have one, so that even in
  // EXPECT_EXACT_RESULTS mode these tests should pass, because 0x7fc00000 is
  // still the exact expected NaN.
  status |= test__subsf3(0x7f800000, 0x7f800000, 0x7fc00000);
  status |= test__subsf3(0xff800000, 0xff800000, 0x7fc00000);
  status |= test__subsf3(0x3f800000, 0x7fc00000, 0x7fc00000);
  status |= test__subsf3(0x7fc00000, 0x3f800000, 0x7fc00000);
  status |= test__subsf3(0x7fc00000, 0x7fc00000, 0x7fc00000);

#ifdef ARM_NAN_HANDLING
  // Tests specific to the NaN handling of Arm hardware, mimicked by the
  // subtraction function in arm/addsf3.S:
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

  status |= test__subsf3(0x00000000, 0x7fad4be3, 0x7fed4be3);
  status |= test__subsf3(0x00000000, 0x7fdf48c7, 0x7fdf48c7);
  status |= test__subsf3(0x00000001, 0x7f970eba, 0x7fd70eba);
  status |= test__subsf3(0x00000001, 0x7fc35716, 0x7fc35716);
  status |= test__subsf3(0x007fffff, 0x7fbf52d6, 0x7fff52d6);
  status |= test__subsf3(0x007fffff, 0x7fc7a2df, 0x7fc7a2df);
  status |= test__subsf3(0x3f800000, 0x7f987a85, 0x7fd87a85);
  status |= test__subsf3(0x3f800000, 0x7fc50124, 0x7fc50124);
  status |= test__subsf3(0x7f7fffff, 0x7f95fd6f, 0x7fd5fd6f);
  status |= test__subsf3(0x7f7fffff, 0x7ffc28dc, 0x7ffc28dc);
  status |= test__subsf3(0x7f800000, 0x7f800000, 0x7fc00000);
  status |= test__subsf3(0x7f800000, 0x7f8dd790, 0x7fcdd790);
  status |= test__subsf3(0x7f800000, 0x7fd2ef2b, 0x7fd2ef2b);
  status |= test__subsf3(0x7f99b09d, 0x00000000, 0x7fd9b09d);
  status |= test__subsf3(0x7f93541e, 0x00000001, 0x7fd3541e);
  status |= test__subsf3(0x7f9fc002, 0x007fffff, 0x7fdfc002);
  status |= test__subsf3(0x7fb5db77, 0x3f800000, 0x7ff5db77);
  status |= test__subsf3(0x7f9f5d92, 0x7f7fffff, 0x7fdf5d92);
  status |= test__subsf3(0x7fac7a36, 0x7f800000, 0x7fec7a36);
  status |= test__subsf3(0x7fb42008, 0x7fb0ee07, 0x7ff42008);
  status |= test__subsf3(0x7f8bd740, 0x7fc7aaf1, 0x7fcbd740);
  status |= test__subsf3(0x7f9bb57b, 0x80000000, 0x7fdbb57b);
  status |= test__subsf3(0x7f951a78, 0x80000001, 0x7fd51a78);
  status |= test__subsf3(0x7f9ba63b, 0x807fffff, 0x7fdba63b);
  status |= test__subsf3(0x7f89463c, 0xbf800000, 0x7fc9463c);
  status |= test__subsf3(0x7fb63563, 0xff7fffff, 0x7ff63563);
  status |= test__subsf3(0x7f90886e, 0xff800000, 0x7fd0886e);
  status |= test__subsf3(0x7fe8c15e, 0x00000000, 0x7fe8c15e);
  status |= test__subsf3(0x7fe915ae, 0x00000001, 0x7fe915ae);
  status |= test__subsf3(0x7ffa9b42, 0x007fffff, 0x7ffa9b42);
  status |= test__subsf3(0x7fdad0f5, 0x3f800000, 0x7fdad0f5);
  status |= test__subsf3(0x7fd10dcb, 0x7f7fffff, 0x7fd10dcb);
  status |= test__subsf3(0x7fd08e8a, 0x7f800000, 0x7fd08e8a);
  status |= test__subsf3(0x7fc3a9e6, 0x7f91a816, 0x7fd1a816);
  status |= test__subsf3(0x7fdb229c, 0x7fc26c68, 0x7fdb229c);
  status |= test__subsf3(0x7fc9f6bb, 0x80000000, 0x7fc9f6bb);
  status |= test__subsf3(0x7ffa178b, 0x80000001, 0x7ffa178b);
  status |= test__subsf3(0x7fef2a0b, 0x807fffff, 0x7fef2a0b);
  status |= test__subsf3(0x7ffc885b, 0xbf800000, 0x7ffc885b);
  status |= test__subsf3(0x7fd26e8c, 0xff7fffff, 0x7fd26e8c);
  status |= test__subsf3(0x7fc55329, 0xff800000, 0x7fc55329);
  status |= test__subsf3(0x80000000, 0x7fa833ae, 0x7fe833ae);
  status |= test__subsf3(0x80000000, 0x7fc4df63, 0x7fc4df63);
  status |= test__subsf3(0x80000001, 0x7f98827d, 0x7fd8827d);
  status |= test__subsf3(0x80000001, 0x7fd7acc5, 0x7fd7acc5);
  status |= test__subsf3(0x807fffff, 0x7fad19c0, 0x7fed19c0);
  status |= test__subsf3(0x807fffff, 0x7ffe1907, 0x7ffe1907);
  status |= test__subsf3(0xbf800000, 0x7fa95487, 0x7fe95487);
  status |= test__subsf3(0xbf800000, 0x7fd2bbee, 0x7fd2bbee);
  status |= test__subsf3(0xff7fffff, 0x7f86ba21, 0x7fc6ba21);
  status |= test__subsf3(0xff7fffff, 0x7feb00d7, 0x7feb00d7);
  status |= test__subsf3(0xff800000, 0x7f857fdc, 0x7fc57fdc);
  status |= test__subsf3(0xff800000, 0x7fde0397, 0x7fde0397);
  status |= test__subsf3(0xff800000, 0xff800000, 0x7fc00000);
#endif // ARM_NAN_HANDLING

  return status;
}
