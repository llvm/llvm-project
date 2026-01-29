// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunsdfsi

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

// Returns: a converted from double to uint32_t
COMPILER_RT_ABI uint32_t __fixunsdfsi(double a);

int test__fixunsdfsi(int line, uint64_t a_rep, uint32_t expected) {
  double a = fromRep64(a_rep);
  int32_t x = __fixunsdfsi(a);
  int ret = x != expected;

  if (ret) {
    printf("error at line %d: __fixunsdfsi(%016" PRIx64 ") = %08" PRIx32
           ", expected %08" PRIx32 "\n",
           line, a_rep, x, expected);
  }
  return ret;
}

#define test__fixunsdfsi(a,x) test__fixunsdfsi(__LINE__,a,x)

int main(void) {
  int status = 0;

  status |= test__fixunsdfsi(0x0000000000000000, 0x00000000);
  status |= test__fixunsdfsi(0x0000000000000001, 0x00000000);
  status |= test__fixunsdfsi(0x001fffefffffffff, 0x00000000);
  status |= test__fixunsdfsi(0x002ffffffbffffff, 0x00000000);
  status |= test__fixunsdfsi(0x380a000000000000, 0x00000000);
  status |= test__fixunsdfsi(0x3fd0000000000000, 0x00000000);
  status |= test__fixunsdfsi(0x3fe0000000000000, 0x00000000);
  status |= test__fixunsdfsi(0x3fe8000000000000, 0x00000000);
  status |= test__fixunsdfsi(0x3fe8000000000000, 0x00000000);
  status |= test__fixunsdfsi(0x3ff0000000000000, 0x00000001);
  status |= test__fixunsdfsi(0x3ff0000000000000, 0x00000001);
  status |= test__fixunsdfsi(0x3ff3cf3a9243d54d, 0x00000001);
  status |= test__fixunsdfsi(0x3ff4000000000000, 0x00000001);
  status |= test__fixunsdfsi(0x3ff8000000000000, 0x00000001);
  status |= test__fixunsdfsi(0x3ff8000000000000, 0x00000001);
  status |= test__fixunsdfsi(0x3ff907591158c5d8, 0x00000001);
  status |= test__fixunsdfsi(0x3ffc000000000000, 0x00000001);
  status |= test__fixunsdfsi(0x4000000000000000, 0x00000002);
  status |= test__fixunsdfsi(0x4000000000000000, 0x00000002);
  status |= test__fixunsdfsi(0x4002000000000000, 0x00000002);
  status |= test__fixunsdfsi(0x4004000000000000, 0x00000002);
  status |= test__fixunsdfsi(0x4004000000000000, 0x00000002);
  status |= test__fixunsdfsi(0x4006000000000000, 0x00000002);
  status |= test__fixunsdfsi(0x4006882d42e373c2, 0x00000002);
  status |= test__fixunsdfsi(0x400af556280c2c53, 0x00000003);
  status |= test__fixunsdfsi(0x400c000000000000, 0x00000003);
  status |= test__fixunsdfsi(0x4010e2652ca3b655, 0x00000004);
  status |= test__fixunsdfsi(0x4013752289364b88, 0x00000004);
  status |= test__fixunsdfsi(0x4018000000000000, 0x00000006);
  status |= test__fixunsdfsi(0x401a000000000000, 0x00000006);
  status |= test__fixunsdfsi(0x401e000000000000, 0x00000007);
  status |= test__fixunsdfsi(0x40213e96e6ee06b6, 0x00000008);
  status |= test__fixunsdfsi(0x4028cdf10f8a4e54, 0x0000000c);
  status |= test__fixunsdfsi(0x402c000000000000, 0x0000000e);
  status |= test__fixunsdfsi(0x402d000000000000, 0x0000000e);
  status |= test__fixunsdfsi(0x402f000000000000, 0x0000000f);
  status |= test__fixunsdfsi(0x4030800000000000, 0x00000010);
  status |= test__fixunsdfsi(0x4034eefd80e0249b, 0x00000014);
  status |= test__fixunsdfsi(0x4037800000000000, 0x00000017);
  status |= test__fixunsdfsi(0x403b000000000000, 0x0000001b);
  status |= test__fixunsdfsi(0x403d6996adec0f09, 0x0000001d);
  status |= test__fixunsdfsi(0x4041d25097b9ee14, 0x00000023);
  status |= test__fixunsdfsi(0x4047c00000000000, 0x0000002f);
  status |= test__fixunsdfsi(0x404b400000000000, 0x00000036);
  status |= test__fixunsdfsi(0x404c0773be0cb9b7, 0x00000038);
  status |= test__fixunsdfsi(0x404e000000000000, 0x0000003c);
  status |= test__fixunsdfsi(0x4051e00000000000, 0x00000047);
  status |= test__fixunsdfsi(0x4053200000000000, 0x0000004c);
  status |= test__fixunsdfsi(0x405589958f279d42, 0x00000056);
  status |= test__fixunsdfsi(0x4059000000000000, 0x00000064);
  status |= test__fixunsdfsi(0x405ea94c1daf1a78, 0x0000007a);
  status |= test__fixunsdfsi(0x40615bb017eb1476, 0x0000008a);
  status |= test__fixunsdfsi(0x4069500000000000, 0x000000ca);
  status |= test__fixunsdfsi(0x406a22674b8b878f, 0x000000d1);
  status |= test__fixunsdfsi(0x406bf00000000000, 0x000000df);
  status |= test__fixunsdfsi(0x406d800000000000, 0x000000ec);
  status |= test__fixunsdfsi(0x4072d80000000000, 0x0000012d);
  status |= test__fixunsdfsi(0x40757c8231fe92f1, 0x00000157);
  status |= test__fixunsdfsi(0x4076a80000000000, 0x0000016a);
  status |= test__fixunsdfsi(0x4077500000000000, 0x00000175);
  status |= test__fixunsdfsi(0x407af61b26e4a441, 0x000001af);
  status |= test__fixunsdfsi(0x4080f40000000000, 0x0000021e);
  status |= test__fixunsdfsi(0x4081363310b2470c, 0x00000226);
  status |= test__fixunsdfsi(0x40860c0000000000, 0x000002c1);
  status |= test__fixunsdfsi(0x408b000000000000, 0x00000360);
  status |= test__fixunsdfsi(0x408e9aaa9a478b59, 0x000003d3);
  status |= test__fixunsdfsi(0x4091c67f05129ed4, 0x00000471);
  status |= test__fixunsdfsi(0x4093a60000000000, 0x000004e9);
  status |= test__fixunsdfsi(0x4098140000000000, 0x00000605);
  status |= test__fixunsdfsi(0x409a5a0000000000, 0x00000696);
  status |= test__fixunsdfsi(0x409ff99df878ad3e, 0x000007fe);
  status |= test__fixunsdfsi(0x40a3500000000000, 0x000009a8);
  status |= test__fixunsdfsi(0x40a5598ffcbb08ba, 0x00000aac);
  status |= test__fixunsdfsi(0x40a956fba09be449, 0x00000cab);
  status |= test__fixunsdfsi(0x40ab8f0000000000, 0x00000dc7);
  status |= test__fixunsdfsi(0x40ad090000000000, 0x00000e84);
  status |= test__fixunsdfsi(0x40b1118000000000, 0x00001111);
  status |= test__fixunsdfsi(0x40b3bab731bb5e6d, 0x000013ba);
  status |= test__fixunsdfsi(0x40b6de0000000000, 0x000016de);
  status |= test__fixunsdfsi(0x40bac06eeb8b97ba, 0x00001ac0);
  status |= test__fixunsdfsi(0x40bce28000000000, 0x00001ce2);
  status |= test__fixunsdfsi(0x40c2870000000000, 0x0000250e);
  status |= test__fixunsdfsi(0x40c84471c85901df, 0x00003088);
  status |= test__fixunsdfsi(0x40c9c34000000000, 0x00003386);
  status |= test__fixunsdfsi(0x40cd9b94b71f57ea, 0x00003b37);
  status |= test__fixunsdfsi(0x40cdc3c000000000, 0x00003b87);
  status |= test__fixunsdfsi(0x40d00da000000000, 0x00004036);
  status |= test__fixunsdfsi(0x40d19f4000000000, 0x0000467d);
  status |= test__fixunsdfsi(0x40d79d704d0443f1, 0x00005e75);
  status |= test__fixunsdfsi(0x40db84e000000000, 0x00006e13);
  status |= test__fixunsdfsi(0x40de81403e6071ea, 0x00007a05);
  status |= test__fixunsdfsi(0x40e2a16f9da2ed87, 0x0000950b);
  status |= test__fixunsdfsi(0x40e92a3000000000, 0x0000c951);
  status |= test__fixunsdfsi(0x40e9d5d000000000, 0x0000ceae);
  status |= test__fixunsdfsi(0x40eb548000000000, 0x0000daa4);
  status |= test__fixunsdfsi(0x40ec19b6638d34af, 0x0000e0cd);
  status |= test__fixunsdfsi(0x40f2d4d49a34df18, 0x00012d4d);
  status |= test__fixunsdfsi(0x40f2de6800000000, 0x00012de6);
  status |= test__fixunsdfsi(0x40f46b9af08e6ece, 0x000146b9);
  status |= test__fixunsdfsi(0x40fb2fe000000000, 0x0001b2fe);
  status |= test__fixunsdfsi(0x40fc81d800000000, 0x0001c81d);
  status |= test__fixunsdfsi(0x4100669800000000, 0x00020cd3);
  status |= test__fixunsdfsi(0x4104a6686f29748d, 0x000294cd);
  status |= test__fixunsdfsi(0x410a1fc576d6489b, 0x000343f8);
  status |= test__fixunsdfsi(0x410b997400000000, 0x0003732e);
  status |= test__fixunsdfsi(0x410e962c00000000, 0x0003d2c5);
  status |= test__fixunsdfsi(0x4113e47a321d351e, 0x0004f91e);
  status |= test__fixunsdfsi(0x41159158c64c86e2, 0x00056456);
  status |= test__fixunsdfsi(0x411ce43e00000000, 0x0007390f);
  status |= test__fixunsdfsi(0x411eacc400000000, 0x0007ab31);
  status |= test__fixunsdfsi(0x411ee00a00000000, 0x0007b802);
  status |= test__fixunsdfsi(0x4120eb1f00000000, 0x0008758f);
  status |= test__fixunsdfsi(0x4121bc002850dcff, 0x0008de00);
  status |= test__fixunsdfsi(0x4123669100000000, 0x0009b348);
  status |= test__fixunsdfsi(0x4125458fefa849cd, 0x000aa2c7);
  status |= test__fixunsdfsi(0x412c5f6600000000, 0x000e2fb3);
  status |= test__fixunsdfsi(0x41311f349fdd064e, 0x00111f34);
  status |= test__fixunsdfsi(0x4135e3c47a5a7295, 0x0015e3c4);
  status |= test__fixunsdfsi(0x413bb95a80000000, 0x001bb95a);
  status |= test__fixunsdfsi(0x413dc4b980000000, 0x001dc4b9);
  status |= test__fixunsdfsi(0x413dded700000000, 0x001dded7);
  status |= test__fixunsdfsi(0x4143339380000000, 0x00266727);
  status |= test__fixunsdfsi(0x4143f42f7838cebe, 0x0027e85e);
  status |= test__fixunsdfsi(0x4148d71240000000, 0x0031ae24);
  status |= test__fixunsdfsi(0x414f8b46986123ff, 0x003f168d);
  status |= test__fixunsdfsi(0x414fc468c0000000, 0x003f88d1);
  status |= test__fixunsdfsi(0x4152d16760000000, 0x004b459d);
  status |= test__fixunsdfsi(0x41559b87ac7fd1fb, 0x00566e1e);
  status |= test__fixunsdfsi(0x415679a847497583, 0x0059e6a1);
  status |= test__fixunsdfsi(0x41568d0e20000000, 0x005a3438);
  status |= test__fixunsdfsi(0x415efb4d80000000, 0x007bed36);
  status |= test__fixunsdfsi(0x41603a2370000000, 0x0081d11b);
  status |= test__fixunsdfsi(0x4160d14709ee668a, 0x00868a38);
  status |= test__fixunsdfsi(0x416705f510000000, 0x00b82fa8);
  status |= test__fixunsdfsi(0x41678a2eb167a88f, 0x00bc5175);
  status |= test__fixunsdfsi(0x416dc05b40000000, 0x00ee02da);
  status |= test__fixunsdfsi(0x41730fb978000000, 0x0130fb97);
  status |= test__fixunsdfsi(0x417395a4a66fca01, 0x01395a4a);
  status |= test__fixunsdfsi(0x41756ef08b6d5dd0, 0x0156ef08);
  status |= test__fixunsdfsi(0x4179efdb00000000, 0x019efdb0);
  status |= test__fixunsdfsi(0x417b4f6208000000, 0x01b4f620);
  status |= test__fixunsdfsi(0x4180907a07f893a5, 0x02120f40);
  status |= test__fixunsdfsi(0x41862857dc000000, 0x02c50afb);
  status |= test__fixunsdfsi(0x4187df63b4000000, 0x02fbec76);
  status |= test__fixunsdfsi(0x418c997fa8000000, 0x03932ff5);
  status |= test__fixunsdfsi(0x418ee2d28aa63b87, 0x03dc5a51);
  status |= test__fixunsdfsi(0x419306468a000000, 0x04c191a2);
  status |= test__fixunsdfsi(0x41948b47dbc198b6, 0x0522d1f6);
  status |= test__fixunsdfsi(0x4195be8a08000000, 0x056fa282);
  status |= test__fixunsdfsi(0x419acb35e46baf44, 0x06b2cd79);
  status |= test__fixunsdfsi(0x419ec43dfe000000, 0x07b10f7f);
  status |= test__fixunsdfsi(0x41a68e6716000000, 0x0b47338b);
  status |= test__fixunsdfsi(0x41a893264f33d251, 0x0c499327);
  status |= test__fixunsdfsi(0x41af11d19d000000, 0x0f88e8ce);
  status |= test__fixunsdfsi(0x41af241394ce98da, 0x0f9209ca);
  status |= test__fixunsdfsi(0x41afb8d0b7000000, 0x0fdc685b);
  status |= test__fixunsdfsi(0x41b1a63370800000, 0x11a63370);
  status |= test__fixunsdfsi(0x41b23df14b800000, 0x123df14b);
  status |= test__fixunsdfsi(0x41b6f2d50149e2d9, 0x16f2d501);
  status |= test__fixunsdfsi(0x41ba1aa592000000, 0x1a1aa592);
  status |= test__fixunsdfsi(0x41bf53402fd53daa, 0x1f53402f);
  status |= test__fixunsdfsi(0x41c18548afc00000, 0x230a915f);
  status |= test__fixunsdfsi(0x41c2e365e5345a6b, 0x25c6cbca);
  status |= test__fixunsdfsi(0x41c4492dac400000, 0x28925b58);
  status |= test__fixunsdfsi(0x41ca895f94000000, 0x3512bf28);
  status |= test__fixunsdfsi(0x41ccc3e5f1e5b560, 0x3987cbe3);
  status |= test__fixunsdfsi(0x41d01a143e200000, 0x406850f8);
  status |= test__fixunsdfsi(0x41d01d7605400000, 0x4075d815);
  status |= test__fixunsdfsi(0x41ddcda3abe00000, 0x77368eaf);
  status |= test__fixunsdfsi(0x41de53dafe34e730, 0x794f6bf8);
  status |= test__fixunsdfsi(0x41df843af68a9ef5, 0x7e10ebda);
  status |= test__fixunsdfsi(0x41e2c728e4400000, 0x96394722);
  status |= test__fixunsdfsi(0x41e950d535f00000, 0xca86a9af);
  status |= test__fixunsdfsi(0x41e9afe08a500000, 0xcd7f0452);
  status |= test__fixunsdfsi(0x41eb81ce4bd25eaa, 0xdc0e725e);
  status |= test__fixunsdfsi(0x41ef6975dbc19d7a, 0xfb4baede);
  status |= test__fixunsdfsi(0x8000000000000000, 0x00000000);
  status |= test__fixunsdfsi(0xb818000000000000, 0x00000000);

#ifdef ARM_INVALID_HANDLING
  // Tests specific to the handling of float-to-integer conversions in
  // Arm hardware, mimicked by arm/fixunsdfsi.S:
  //
  //  - too-large positive inputs, including +infinity, return the
  //    maximum possible unsigned integer value
  //
  //  - negative inputs too small to round up to 0, including
  //    -infinity, return 0
  //
  //  - NaN inputs return 0
  status |= test__fixunsdfsi(0x41efffffffffffff, 0xffffffff);
  status |= test__fixunsdfsi(0x41f0000000000000, 0xffffffff);
  status |= test__fixunsdfsi(0x41f37bc9c8400000, 0xffffffff);
  status |= test__fixunsdfsi(0x41f3c0b771e5a126, 0xffffffff);
  status |= test__fixunsdfsi(0x41fb837587480000, 0xffffffff);
  status |= test__fixunsdfsi(0x41fc069b87f80000, 0xffffffff);
  status |= test__fixunsdfsi(0x41feea6325bf9a55, 0xffffffff);
  status |= test__fixunsdfsi(0x7ff0000000000000, 0xffffffff);
  status |= test__fixunsdfsi(0x7ff6d1ebdfe15ee3, 0x00000000);
  status |= test__fixunsdfsi(0x7ff9a4da74944a09, 0x00000000);
  status |= test__fixunsdfsi(0xbfefffffffffffff, 0x00000000);
  status |= test__fixunsdfsi(0xbff0000000000000, 0x00000000);
  status |= test__fixunsdfsi(0xc000000000000000, 0x00000000);
  status |= test__fixunsdfsi(0xfff0000000000000, 0x00000000);

#endif // ARM_INVALID_HANDLING

  return status;
}
