// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_truncdfsf2

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

// Returns: a converted from double to float
COMPILER_RT_ABI float __truncdfsf2(double a);

int test__truncdfsf2(int line, uint64_t a_rep, uint32_t expected_rep) {
  double a = fromRep64(a_rep);
  float x = __truncdfsf2(a);
#ifdef EXPECT_EXACT_RESULTS
  int ret = toRep32(x) != expected_rep;
#else
  int ret = compareResultF(x, expected_rep);
#endif

  if (ret) {
    printf("error at line %d: __truncdfsf2(%016" PRIx64 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a_rep, toRep32(x), expected_rep);
  }
  return ret;
}

#define test__truncdfsf2(a,x) test__truncdfsf2(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__truncdfsf2(0x0000000000000001, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000002, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000004, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000008, 0x00000000);
  status |= test__truncdfsf2(0x000000000000001a, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000020, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000040, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000080, 0x00000000);
  status |= test__truncdfsf2(0x000000000000019a, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000200, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000400, 0x00000000);
  status |= test__truncdfsf2(0x0000000000000800, 0x00000000);
  status |= test__truncdfsf2(0x000000000000189a, 0x00000000);
  status |= test__truncdfsf2(0x0000000000002000, 0x00000000);
  status |= test__truncdfsf2(0x0000000000004000, 0x00000000);
  status |= test__truncdfsf2(0x0000000000008000, 0x00000000);
  status |= test__truncdfsf2(0x000000000001789a, 0x00000000);
  status |= test__truncdfsf2(0x0000000000020000, 0x00000000);
  status |= test__truncdfsf2(0x0000000000040000, 0x00000000);
  status |= test__truncdfsf2(0x0000000000080000, 0x00000000);
  status |= test__truncdfsf2(0x000000000016789a, 0x00000000);
  status |= test__truncdfsf2(0x0000000000200000, 0x00000000);
  status |= test__truncdfsf2(0x0000000000400000, 0x00000000);
  status |= test__truncdfsf2(0x0000000000800000, 0x00000000);
  status |= test__truncdfsf2(0x000000000156789a, 0x00000000);
  status |= test__truncdfsf2(0x0000000002000000, 0x00000000);
  status |= test__truncdfsf2(0x0000000004000000, 0x00000000);
  status |= test__truncdfsf2(0x0000000008000000, 0x00000000);
  status |= test__truncdfsf2(0x000000001456789a, 0x00000000);
  status |= test__truncdfsf2(0x0000000020000000, 0x00000000);
  status |= test__truncdfsf2(0x0000000040000000, 0x00000000);
  status |= test__truncdfsf2(0x0000000080000000, 0x00000000);
  status |= test__truncdfsf2(0x000000013465789a, 0x00000000);
  status |= test__truncdfsf2(0x0000000200000000, 0x00000000);
  status |= test__truncdfsf2(0x0000000400000000, 0x00000000);
  status |= test__truncdfsf2(0x0000000800000000, 0x00000000);
  status |= test__truncdfsf2(0x000000123456789a, 0x00000000);
  status |= test__truncdfsf2(0x0000002000000000, 0x00000000);
  status |= test__truncdfsf2(0x0000004000000000, 0x00000000);
  status |= test__truncdfsf2(0x0000008000000000, 0x00000000);
  status |= test__truncdfsf2(0x000001123456789a, 0x00000000);
  status |= test__truncdfsf2(0x0000020000000000, 0x00000000);
  status |= test__truncdfsf2(0x0000040000000000, 0x00000000);
  status |= test__truncdfsf2(0x0000080000000000, 0x00000000);
  status |= test__truncdfsf2(0x000010123456789a, 0x00000000);
  status |= test__truncdfsf2(0x0000200000000000, 0x00000000);
  status |= test__truncdfsf2(0x0000400000000000, 0x00000000);
  status |= test__truncdfsf2(0x0000800000000000, 0x00000000);
  status |= test__truncdfsf2(0x000100123456789a, 0x00000000);
  status |= test__truncdfsf2(0x0002000000000000, 0x00000000);
  status |= test__truncdfsf2(0x0004000000000000, 0x00000000);
  status |= test__truncdfsf2(0x0008000000000000, 0x00000000);
  status |= test__truncdfsf2(0x0010000000000000, 0x00000000);
  status |= test__truncdfsf2(0x36a0000000000000, 0x00000001);
  status |= test__truncdfsf2(0x36b0000000000000, 0x00000002);
  status |= test__truncdfsf2(0x36b2000000000000, 0x00000002);
  status |= test__truncdfsf2(0x36b4000000000000, 0x00000002);
  status |= test__truncdfsf2(0x36b6000000000000, 0x00000003);
  status |= test__truncdfsf2(0x36b8000000000000, 0x00000003);
  status |= test__truncdfsf2(0x36ba000000000000, 0x00000003);
  status |= test__truncdfsf2(0x36bc000000000000, 0x00000004);
  status |= test__truncdfsf2(0x36be000000000000, 0x00000004);
  status |= test__truncdfsf2(0x36c0000000000000, 0x00000004);
  status |= test__truncdfsf2(0x36c1000000000000, 0x00000004);
  status |= test__truncdfsf2(0x36c2000000000000, 0x00000004);
  status |= test__truncdfsf2(0x36c3000000000000, 0x00000005);
  status |= test__truncdfsf2(0x36c4000000000000, 0x00000005);
  status |= test__truncdfsf2(0x36c5000000000000, 0x00000005);
  status |= test__truncdfsf2(0x36c6000000000000, 0x00000006);
  status |= test__truncdfsf2(0x36c7000000000000, 0x00000006);
  status |= test__truncdfsf2(0x36d0000000000000, 0x00000008);
  status |= test__truncdfsf2(0x36d0800000000000, 0x00000008);
  status |= test__truncdfsf2(0x36d1000000000000, 0x00000008);
  status |= test__truncdfsf2(0x36d1800000000000, 0x00000009);
  status |= test__truncdfsf2(0x36d2000000000000, 0x00000009);
  status |= test__truncdfsf2(0x36d2800000000000, 0x00000009);
  status |= test__truncdfsf2(0x36d3000000000000, 0x0000000a);
  status |= test__truncdfsf2(0x36d3800000000000, 0x0000000a);
  status |= test__truncdfsf2(0x36e0000000000000, 0x00000010);
  status |= test__truncdfsf2(0x36e0400000000000, 0x00000010);
  status |= test__truncdfsf2(0x36e0800000000000, 0x00000010);
  status |= test__truncdfsf2(0x36e0c00000000000, 0x00000011);
  status |= test__truncdfsf2(0x36e1000000000000, 0x00000011);
  status |= test__truncdfsf2(0x36e1400000000000, 0x00000011);
  status |= test__truncdfsf2(0x36e1800000000000, 0x00000012);
  status |= test__truncdfsf2(0x36e1c00000000000, 0x00000012);
  status |= test__truncdfsf2(0x36f0000000000000, 0x00000020);
  status |= test__truncdfsf2(0x36f0200000000000, 0x00000020);
  status |= test__truncdfsf2(0x36f0400000000000, 0x00000020);
  status |= test__truncdfsf2(0x36f0600000000000, 0x00000021);
  status |= test__truncdfsf2(0x36f0800000000000, 0x00000021);
  status |= test__truncdfsf2(0x36f0a00000000000, 0x00000021);
  status |= test__truncdfsf2(0x36f0c00000000000, 0x00000022);
  status |= test__truncdfsf2(0x36f0e00000000000, 0x00000022);
  status |= test__truncdfsf2(0x3700000000000000, 0x00000040);
  status |= test__truncdfsf2(0x3700100000000000, 0x00000040);
  status |= test__truncdfsf2(0x3700200000000000, 0x00000040);
  status |= test__truncdfsf2(0x3700300000000000, 0x00000041);
  status |= test__truncdfsf2(0x3700400000000000, 0x00000041);
  status |= test__truncdfsf2(0x3700500000000000, 0x00000041);
  status |= test__truncdfsf2(0x3700600000000000, 0x00000042);
  status |= test__truncdfsf2(0x3700700000000000, 0x00000042);
  status |= test__truncdfsf2(0x3710000000000000, 0x00000080);
  status |= test__truncdfsf2(0x3710080000000000, 0x00000080);
  status |= test__truncdfsf2(0x3710100000000000, 0x00000080);
  status |= test__truncdfsf2(0x3710180000000000, 0x00000081);
  status |= test__truncdfsf2(0x3710200000000000, 0x00000081);
  status |= test__truncdfsf2(0x3710280000000000, 0x00000081);
  status |= test__truncdfsf2(0x3710300000000000, 0x00000082);
  status |= test__truncdfsf2(0x3710380000000000, 0x00000082);
  status |= test__truncdfsf2(0x3720000000000000, 0x00000100);
  status |= test__truncdfsf2(0x3720040000000000, 0x00000100);
  status |= test__truncdfsf2(0x3720080000000000, 0x00000100);
  status |= test__truncdfsf2(0x37200c0000000000, 0x00000101);
  status |= test__truncdfsf2(0x3720100000000000, 0x00000101);
  status |= test__truncdfsf2(0x3720140000000000, 0x00000101);
  status |= test__truncdfsf2(0x3720180000000000, 0x00000102);
  status |= test__truncdfsf2(0x37201c0000000000, 0x00000102);
  status |= test__truncdfsf2(0x3730000000000000, 0x00000200);
  status |= test__truncdfsf2(0x3730020000000000, 0x00000200);
  status |= test__truncdfsf2(0x3730040000000000, 0x00000200);
  status |= test__truncdfsf2(0x3730060000000000, 0x00000201);
  status |= test__truncdfsf2(0x3730080000000000, 0x00000201);
  status |= test__truncdfsf2(0x37300a0000000000, 0x00000201);
  status |= test__truncdfsf2(0x37300c0000000000, 0x00000202);
  status |= test__truncdfsf2(0x37300e0000000000, 0x00000202);
  status |= test__truncdfsf2(0x3740000000000000, 0x00000400);
  status |= test__truncdfsf2(0x3740010000000000, 0x00000400);
  status |= test__truncdfsf2(0x3740020000000000, 0x00000400);
  status |= test__truncdfsf2(0x3740030000000000, 0x00000401);
  status |= test__truncdfsf2(0x3740040000000000, 0x00000401);
  status |= test__truncdfsf2(0x3740050000000000, 0x00000401);
  status |= test__truncdfsf2(0x3740060000000000, 0x00000402);
  status |= test__truncdfsf2(0x3740070000000000, 0x00000402);
  status |= test__truncdfsf2(0x3750000000000000, 0x00000800);
  status |= test__truncdfsf2(0x3750008000000000, 0x00000800);
  status |= test__truncdfsf2(0x3750010000000000, 0x00000800);
  status |= test__truncdfsf2(0x3750018000000000, 0x00000801);
  status |= test__truncdfsf2(0x3750020000000000, 0x00000801);
  status |= test__truncdfsf2(0x3750028000000000, 0x00000801);
  status |= test__truncdfsf2(0x3750030000000000, 0x00000802);
  status |= test__truncdfsf2(0x3750038000000000, 0x00000802);
  status |= test__truncdfsf2(0x3760000000000000, 0x00001000);
  status |= test__truncdfsf2(0x3760004000000000, 0x00001000);
  status |= test__truncdfsf2(0x3760008000000000, 0x00001000);
  status |= test__truncdfsf2(0x376000c000000000, 0x00001001);
  status |= test__truncdfsf2(0x3760010000000000, 0x00001001);
  status |= test__truncdfsf2(0x3760014000000000, 0x00001001);
  status |= test__truncdfsf2(0x3760018000000000, 0x00001002);
  status |= test__truncdfsf2(0x376001c000000000, 0x00001002);
  status |= test__truncdfsf2(0x3770000000000000, 0x00002000);
  status |= test__truncdfsf2(0x3770002000000000, 0x00002000);
  status |= test__truncdfsf2(0x3770004000000000, 0x00002000);
  status |= test__truncdfsf2(0x3770006000000000, 0x00002001);
  status |= test__truncdfsf2(0x3770008000000000, 0x00002001);
  status |= test__truncdfsf2(0x377000a000000000, 0x00002001);
  status |= test__truncdfsf2(0x377000c000000000, 0x00002002);
  status |= test__truncdfsf2(0x377000e000000000, 0x00002002);
  status |= test__truncdfsf2(0x3780000000000000, 0x00004000);
  status |= test__truncdfsf2(0x3780001000000000, 0x00004000);
  status |= test__truncdfsf2(0x3780002000000000, 0x00004000);
  status |= test__truncdfsf2(0x3780003000000000, 0x00004001);
  status |= test__truncdfsf2(0x3780004000000000, 0x00004001);
  status |= test__truncdfsf2(0x3780005000000000, 0x00004001);
  status |= test__truncdfsf2(0x3780006000000000, 0x00004002);
  status |= test__truncdfsf2(0x3780007000000000, 0x00004002);
  status |= test__truncdfsf2(0x3790000000000000, 0x00008000);
  status |= test__truncdfsf2(0x3790000800000000, 0x00008000);
  status |= test__truncdfsf2(0x3790001000000000, 0x00008000);
  status |= test__truncdfsf2(0x3790001800000000, 0x00008001);
  status |= test__truncdfsf2(0x3790002000000000, 0x00008001);
  status |= test__truncdfsf2(0x3790002800000000, 0x00008001);
  status |= test__truncdfsf2(0x3790003000000000, 0x00008002);
  status |= test__truncdfsf2(0x3790003800000000, 0x00008002);
  status |= test__truncdfsf2(0x37a0000000000000, 0x00010000);
  status |= test__truncdfsf2(0x37a0000400000000, 0x00010000);
  status |= test__truncdfsf2(0x37a0000800000000, 0x00010000);
  status |= test__truncdfsf2(0x37a0000c00000000, 0x00010001);
  status |= test__truncdfsf2(0x37a0001000000000, 0x00010001);
  status |= test__truncdfsf2(0x37a0001400000000, 0x00010001);
  status |= test__truncdfsf2(0x37a0001800000000, 0x00010002);
  status |= test__truncdfsf2(0x37a0001c00000000, 0x00010002);
  status |= test__truncdfsf2(0x37b0000000000000, 0x00020000);
  status |= test__truncdfsf2(0x37b0000200000000, 0x00020000);
  status |= test__truncdfsf2(0x37b0000400000000, 0x00020000);
  status |= test__truncdfsf2(0x37b0000600000000, 0x00020001);
  status |= test__truncdfsf2(0x37b0000800000000, 0x00020001);
  status |= test__truncdfsf2(0x37b0000a00000000, 0x00020001);
  status |= test__truncdfsf2(0x37b0000c00000000, 0x00020002);
  status |= test__truncdfsf2(0x37b0000e00000000, 0x00020002);
  status |= test__truncdfsf2(0x37c0000000000000, 0x00040000);
  status |= test__truncdfsf2(0x37c0000100000000, 0x00040000);
  status |= test__truncdfsf2(0x37c0000200000000, 0x00040000);
  status |= test__truncdfsf2(0x37c0000300000000, 0x00040001);
  status |= test__truncdfsf2(0x37c0000400000000, 0x00040001);
  status |= test__truncdfsf2(0x37c0000500000000, 0x00040001);
  status |= test__truncdfsf2(0x37c0000600000000, 0x00040002);
  status |= test__truncdfsf2(0x37c0000700000000, 0x00040002);
  status |= test__truncdfsf2(0x37d0000000000000, 0x00080000);
  status |= test__truncdfsf2(0x37d0000080000000, 0x00080000);
  status |= test__truncdfsf2(0x37d0000100000000, 0x00080000);
  status |= test__truncdfsf2(0x37d0000180000000, 0x00080001);
  status |= test__truncdfsf2(0x37d0000200000000, 0x00080001);
  status |= test__truncdfsf2(0x37d0000280000000, 0x00080001);
  status |= test__truncdfsf2(0x37d0000300000000, 0x00080002);
  status |= test__truncdfsf2(0x37d0000380000000, 0x00080002);
  status |= test__truncdfsf2(0x37e0000000000000, 0x00100000);
  status |= test__truncdfsf2(0x37e0000040000000, 0x00100000);
  status |= test__truncdfsf2(0x37e0000080000000, 0x00100000);
  status |= test__truncdfsf2(0x37e00000c0000000, 0x00100001);
  status |= test__truncdfsf2(0x37e0000100000000, 0x00100001);
  status |= test__truncdfsf2(0x37e0000140000000, 0x00100001);
  status |= test__truncdfsf2(0x37e0000180000000, 0x00100002);
  status |= test__truncdfsf2(0x37e00001c0000000, 0x00100002);
  status |= test__truncdfsf2(0x37f0000000000000, 0x00200000);
  status |= test__truncdfsf2(0x37f0000020000000, 0x00200000);
  status |= test__truncdfsf2(0x37f000003fffffff, 0x00200000);
  status |= test__truncdfsf2(0x37f0000040000000, 0x00200000);
  status |= test__truncdfsf2(0x37f0000040000001, 0x00200001);
  status |= test__truncdfsf2(0x37f0000060000000, 0x00200001);
  status |= test__truncdfsf2(0x37f0000080000000, 0x00200001);
  status |= test__truncdfsf2(0x37f00000a0000000, 0x00200001);
  status |= test__truncdfsf2(0x37f00000bfffffff, 0x00200001);
  status |= test__truncdfsf2(0x37f00000c0000000, 0x00200002);
  status |= test__truncdfsf2(0x37f00000c0000001, 0x00200002);
  status |= test__truncdfsf2(0x37f00000e0000000, 0x00200002);
  status |= test__truncdfsf2(0x3800000000000000, 0x00400000);
  status |= test__truncdfsf2(0x3800000010000000, 0x00400000);
  status |= test__truncdfsf2(0x3800000020000000, 0x00400000);
  status |= test__truncdfsf2(0x3800000030000000, 0x00400001);
  status |= test__truncdfsf2(0x3800000040000000, 0x00400001);
  status |= test__truncdfsf2(0x3800000050000000, 0x00400001);
  status |= test__truncdfsf2(0x3800000060000000, 0x00400002);
  status |= test__truncdfsf2(0x3800000070000000, 0x00400002);
  status |= test__truncdfsf2(0x380fffffffffffff, 0x00800000);
  status |= test__truncdfsf2(0x3810000000000000, 0x00800000);
  status |= test__truncdfsf2(0x3810000008000000, 0x00800000);
  status |= test__truncdfsf2(0x3810000010000000, 0x00800000);
  status |= test__truncdfsf2(0x3810000018000000, 0x00800001);
  status |= test__truncdfsf2(0x3810000020000000, 0x00800001);
  status |= test__truncdfsf2(0x3810000028000000, 0x00800001);
  status |= test__truncdfsf2(0x3810000030000000, 0x00800002);
  status |= test__truncdfsf2(0x3810000038000000, 0x00800002);
  status |= test__truncdfsf2(0x3ff0000000000000, 0x3f800000);
  status |= test__truncdfsf2(0x3ff0000008000000, 0x3f800000);
  status |= test__truncdfsf2(0x3ff0000010000000, 0x3f800000);
  status |= test__truncdfsf2(0x3ff0000018000000, 0x3f800001);
  status |= test__truncdfsf2(0x3ff0000028000000, 0x3f800001);
  status |= test__truncdfsf2(0x3ff0000030000000, 0x3f800002);
  status |= test__truncdfsf2(0x3ff0000038000000, 0x3f800002);
  status |= test__truncdfsf2(0x4000000000000000, 0x40000000);
  status |= test__truncdfsf2(0x47efffffe8000000, 0x7f7fffff);
  status |= test__truncdfsf2(0x47effffff0000000, 0x7f800000);
  status |= test__truncdfsf2(0x47effffff8000000, 0x7f800000);
  status |= test__truncdfsf2(0x7fc0000000000000, 0x7f800000);
  status |= test__truncdfsf2(0x7ff0000000000000, 0x7f800000);
  status |= test__truncdfsf2(0x8010000000000000, 0x80000000);
  status |= test__truncdfsf2(0xbff0000008000000, 0xbf800000);
  status |= test__truncdfsf2(0xbff0000010000000, 0xbf800000);
  status |= test__truncdfsf2(0xbff0000018000000, 0xbf800001);
  status |= test__truncdfsf2(0xbff0000028000000, 0xbf800001);
  status |= test__truncdfsf2(0xbff0000030000000, 0xbf800002);
  status |= test__truncdfsf2(0xbff0000038000000, 0xbf800002);
  status |= test__truncdfsf2(0xc024000000000000, 0xc1200000);
  status |= test__truncdfsf2(0xc7efffffe8000000, 0xff7fffff);
  status |= test__truncdfsf2(0xc7effffff0000000, 0xff800000);
  status |= test__truncdfsf2(0xc7effffff8000000, 0xff800000);
  status |= test__truncdfsf2(0xffc0000000000000, 0xff800000);
  status |= test__truncdfsf2(0xfff0000000000000, 0xff800000);
  status |= test__truncdfsf2(0x3780000000000000, 0x00004000);
  status |= test__truncdfsf2(0xb780000000000000, 0x80004000);
  status |= test__truncdfsf2(0x0000000080000000, 0x00000000);
  status |= test__truncdfsf2(0x8000000080000000, 0x80000000);
  status |= test__truncdfsf2(0x380ffffff0000000, 0x00800000);
  status |= test__truncdfsf2(0x380fffffd0000000, 0x007fffff);
  status |= test__truncdfsf2(0x380fffffe8000000, 0x00800000);
  status |= test__truncdfsf2(0x380fffffc8000000, 0x007fffff);
  status |= test__truncdfsf2(0xb80ffffff0000000, 0x80800000);
  status |= test__truncdfsf2(0xb80fffffd0000000, 0x807fffff);
  status |= test__truncdfsf2(0xb80fffffe8000000, 0x80800000);
  status |= test__truncdfsf2(0xb80fffffc8000000, 0x807fffff);
  status |= test__truncdfsf2(0x0000000000000000, 0x00000000);
  status |= test__truncdfsf2(0x8000000000000000, 0x80000000);
  status |= test__truncdfsf2(0xc7e0000010000000, 0xff000000);

  // Test that the result of an operation is a NaN at all when it should be.
  //
  // In most configurations these tests' results are checked compared using
  // compareResultF, so we set all the answers to the canonical NaN 0x7fc00000,
  // which causes compareResultF to accept any NaN encoding. We also use the
  // same value as the input NaN in tests that have one, so that even in
  // EXPECT_EXACT_RESULTS mode these tests should pass, because 0x7fc00000 is
  // still the exact expected NaN.
  status |= test__truncdfsf2(0x7ff8000000000000, 0x7fc00000);

#ifdef ARM_NAN_HANDLING
  // Tests specific to the NaN handling of Arm hardware, mimicked by
  // arm/truncdfsf2.S:
  //
  //  - a quiet NaN is distinguished by the top mantissa bit being 1
  //
  //  - converting a quiet NaN from double to float is done by keeping
  //    the topmost 23 bits of the mantissa and discarding the lower
  //    ones
  //
  //  - if the input is a signalling NaN, its top mantissa bit is set
  //    to turn it quiet, and then that quiet NaN is converted to
  //    float as above
  status |= test__truncdfsf2(0x7ff0000000000001, 0x7fc00000);
  status |= test__truncdfsf2(0x7ff753b1887bcf03, 0x7ffa9d8c);
  status |= test__truncdfsf2(0x7ff911d3c0abfdda, 0x7fc88e9e);
  status |= test__truncdfsf2(0xfff0000000000001, 0xffc00000);
  status |= test__truncdfsf2(0xfff753b1887bcf03, 0xfffa9d8c);
  status |= test__truncdfsf2(0xfff911d3c0abfdda, 0xffc88e9e);

#endif // ARM_NAN_HANDLING

  return status;
}
