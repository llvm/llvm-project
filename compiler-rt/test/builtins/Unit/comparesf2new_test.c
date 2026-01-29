// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_comparesf2

#include "int_lib.h"
#include <inttypes.h>
#include <stdio.h>

#include "fp_test.h"

COMPILER_RT_ABI int __eqsf2(float, float);
COMPILER_RT_ABI int __nesf2(float, float);
COMPILER_RT_ABI int __gesf2(float, float);
COMPILER_RT_ABI int __gtsf2(float, float);
COMPILER_RT_ABI int __lesf2(float, float);
COMPILER_RT_ABI int __ltsf2(float, float);
COMPILER_RT_ABI int __cmpsf2(float, float);
COMPILER_RT_ABI int __unordsf2(float, float);

enum Result {
  RESULT_LT,
  RESULT_GT,
  RESULT_EQ,
  RESULT_UN
};

int expect(int line, uint32_t a_rep, uint32_t b_rep, const char *name, int result, int ok, const char *expected) {
  if (!ok)
    printf("error at line %d: %s(%08" PRIx32 ", %08" PRIx32 ") = %d, expected %s\n",
           line, name, a_rep, b_rep, result, expected);
  return !ok;
}

int test__comparesf2(int line, uint32_t a_rep, uint32_t b_rep, enum Result result) {
  float a = fromRep32(a_rep), b = fromRep32(b_rep);

  int eq = __eqsf2(a, b);
  int ne = __nesf2(a, b);
  int ge = __gesf2(a, b);
  int gt = __gtsf2(a, b);
  int le = __lesf2(a, b);
  int lt = __ltsf2(a, b);
  int cmp = __cmpsf2(a, b);
  int unord = __unordsf2(a, b);

  int ret = 0;

  switch (result) {
  case RESULT_LT:
    ret |= expect(line, a_rep, b_rep, "__eqsf2", eq, eq != 0, "!= 0");
    ret |= expect(line, a_rep, b_rep, "__nesf2", ne, ne != 0, "!= 0");
    ret |= expect(line, a_rep, b_rep, "__gesf2", ge, ge < 0, "< 0");
    ret |= expect(line, a_rep, b_rep, "__gtsf2", gt, gt <= 0, "<= 0");
    ret |= expect(line, a_rep, b_rep, "__lesf2", le, le <= 0, "<= 0");
    ret |= expect(line, a_rep, b_rep, "__ltsf2", lt, lt < 0, "< 0");
    ret |= expect(line, a_rep, b_rep, "__cmpsf2", cmp, cmp == -1, "== -1");
    ret |= expect(line, a_rep, b_rep, "__unordsf2", unord, unord == 0, "== 0");
    break;
  case RESULT_GT:
    ret |= expect(line, a_rep, b_rep, "__eqsf2", eq, eq != 0, "!= 0");
    ret |= expect(line, a_rep, b_rep, "__nesf2", ne, ne != 0, "!= 0");
    ret |= expect(line, a_rep, b_rep, "__gesf2", ge, ge >= 0, ">= 0");
    ret |= expect(line, a_rep, b_rep, "__gtsf2", gt, gt > 0, "> 0");
    ret |= expect(line, a_rep, b_rep, "__lesf2", le, le > 0, "> 0");
    ret |= expect(line, a_rep, b_rep, "__ltsf2", lt, lt >= 0, ">= 0");
    ret |= expect(line, a_rep, b_rep, "__cmpsf2", cmp, cmp == 1, "== 1");
    ret |= expect(line, a_rep, b_rep, "__unordsf2", unord, unord == 0, "== 0");
    break;
  case RESULT_EQ:
    ret |= expect(line, a_rep, b_rep, "__eqsf2", eq, eq == 0, "== 0");
    ret |= expect(line, a_rep, b_rep, "__nesf2", ne, ne == 0, "== 0");
    ret |= expect(line, a_rep, b_rep, "__gesf2", ge, ge >= 0, ">= 0");
    ret |= expect(line, a_rep, b_rep, "__gtsf2", gt, gt <= 0, "<= 0");
    ret |= expect(line, a_rep, b_rep, "__lesf2", le, le <= 0, "<= 0");
    ret |= expect(line, a_rep, b_rep, "__ltsf2", lt, lt >= 0, ">= 0");
    ret |= expect(line, a_rep, b_rep, "__cmpsf2", cmp, cmp == 0, "== 0");
    ret |= expect(line, a_rep, b_rep, "__unordsf2", unord, unord == 0, "== 0");
    break;
  case RESULT_UN:
    ret |= expect(line, a_rep, b_rep, "__eqsf2", eq, eq != 0, "!= 0");
    ret |= expect(line, a_rep, b_rep, "__nesf2", ne, ne != 0, "!= 0");
    ret |= expect(line, a_rep, b_rep, "__gesf2", ge, ge < 0, "< 0");
    ret |= expect(line, a_rep, b_rep, "__gtsf2", gt, gt <= 0, "<= 0");
    ret |= expect(line, a_rep, b_rep, "__lesf2", le, le > 0, "> 0");
    ret |= expect(line, a_rep, b_rep, "__ltsf2", lt, lt >= 0, ">= 0");
    ret |= expect(line, a_rep, b_rep, "__cmpsf2", cmp, cmp == 1, "== 1");
    ret |= expect(line, a_rep, b_rep, "__unordsf2", unord, unord == 1, "== 1");
    break;
  }

  return ret;
}

#define test__comparesf2(a,b,x) test__comparesf2(__LINE__,a,b,x)

int main(void) {
  int status = 0;

  status |= test__comparesf2(0x00000000, 0x00000001, RESULT_LT);
  status |= test__comparesf2(0x00000000, 0x007fffff, RESULT_LT);
  status |= test__comparesf2(0x00000000, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0x00000000, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0x00000000, 0x7f800000, RESULT_LT);
  status |= test__comparesf2(0x00000000, 0x7f872da0, RESULT_UN);
  status |= test__comparesf2(0x00000000, 0x7fe42e09, RESULT_UN);
  status |= test__comparesf2(0x00000000, 0x80000000, RESULT_EQ);
  status |= test__comparesf2(0x00000000, 0x80000001, RESULT_GT);
  status |= test__comparesf2(0x00000000, 0x807fffff, RESULT_GT);
  status |= test__comparesf2(0x00000000, 0x80800000, RESULT_GT);
  status |= test__comparesf2(0x00000000, 0xff800000, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0x00000001, RESULT_EQ);
  status |= test__comparesf2(0x00000001, 0x3f7fffff, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x3ffffffe, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x3fffffff, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x7effffff, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x7f7ffffe, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x7f7fffff, RESULT_LT);
  status |= test__comparesf2(0x00000001, 0x7f94d5b9, RESULT_UN);
  status |= test__comparesf2(0x00000001, 0x7fef53b1, RESULT_UN);
  status |= test__comparesf2(0x00000001, 0x80000001, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xbf7fffff, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xbffffffe, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xbfffffff, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xfeffffff, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xff7ffffe, RESULT_GT);
  status |= test__comparesf2(0x00000001, 0xff7fffff, RESULT_GT);
  status |= test__comparesf2(0x00000002, 0x00000001, RESULT_GT);
  status |= test__comparesf2(0x00000003, 0x00000002, RESULT_GT);
  status |= test__comparesf2(0x00000003, 0x40400000, RESULT_LT);
  status |= test__comparesf2(0x00000003, 0x40a00000, RESULT_LT);
  status |= test__comparesf2(0x00000003, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0x00000003, 0xc0a00000, RESULT_GT);
  status |= test__comparesf2(0x00000003, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x00000004, 0x00000004, RESULT_EQ);
  status |= test__comparesf2(0x007ffffc, 0x807ffffc, RESULT_GT);
  status |= test__comparesf2(0x007ffffd, 0x007ffffe, RESULT_LT);
  status |= test__comparesf2(0x007fffff, 0x00000000, RESULT_GT);
  status |= test__comparesf2(0x007fffff, 0x007ffffe, RESULT_GT);
  status |= test__comparesf2(0x007fffff, 0x007fffff, RESULT_EQ);
  status |= test__comparesf2(0x007fffff, 0x00800000, RESULT_LT);
  status |= test__comparesf2(0x007fffff, 0x7f800000, RESULT_LT);
  status |= test__comparesf2(0x007fffff, 0x7fa111d3, RESULT_UN);
  status |= test__comparesf2(0x007fffff, 0x7ff43134, RESULT_UN);
  status |= test__comparesf2(0x007fffff, 0x80000000, RESULT_GT);
  status |= test__comparesf2(0x007fffff, 0xff800000, RESULT_GT);
  status |= test__comparesf2(0x00800000, 0x00000000, RESULT_GT);
  status |= test__comparesf2(0x00800000, 0x00800000, RESULT_EQ);
  status |= test__comparesf2(0x00800000, 0x80800000, RESULT_GT);
  status |= test__comparesf2(0x00800001, 0x00800000, RESULT_GT);
  status |= test__comparesf2(0x00800001, 0x00800002, RESULT_LT);
  status |= test__comparesf2(0x00ffffff, 0x01000000, RESULT_LT);
  status |= test__comparesf2(0x00ffffff, 0x01000002, RESULT_LT);
  status |= test__comparesf2(0x00ffffff, 0x01000004, RESULT_LT);
  status |= test__comparesf2(0x01000000, 0x00ffffff, RESULT_GT);
  status |= test__comparesf2(0x01000001, 0x00800001, RESULT_GT);
  status |= test__comparesf2(0x01000001, 0x00ffffff, RESULT_GT);
  status |= test__comparesf2(0x01000002, 0x00800001, RESULT_GT);
  status |= test__comparesf2(0x017fffff, 0x01800000, RESULT_LT);
  status |= test__comparesf2(0x01800000, 0x017fffff, RESULT_GT);
  status |= test__comparesf2(0x01800001, 0x017fffff, RESULT_GT);
  status |= test__comparesf2(0x01800002, 0x01000003, RESULT_GT);
  status |= test__comparesf2(0x3f000000, 0x3f000000, RESULT_EQ);
  status |= test__comparesf2(0x3f7fffff, 0x00000001, RESULT_GT);
  status |= test__comparesf2(0x3f7fffff, 0x80000001, RESULT_GT);
  status |= test__comparesf2(0x3f800000, 0x3f800000, RESULT_EQ);
  status |= test__comparesf2(0x3f800000, 0x3f800003, RESULT_LT);
  status |= test__comparesf2(0x3f800000, 0x40000000, RESULT_LT);
  status |= test__comparesf2(0x3f800000, 0x40e00000, RESULT_LT);
  status |= test__comparesf2(0x3f800000, 0x7fb27f62, RESULT_UN);
  status |= test__comparesf2(0x3f800000, 0x7fd9d4b4, RESULT_UN);
  status |= test__comparesf2(0x3f800000, 0x80000000, RESULT_GT);
  status |= test__comparesf2(0x3f800000, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x3f800000, 0xbf800003, RESULT_GT);
  status |= test__comparesf2(0x3f800001, 0x3f800000, RESULT_GT);
  status |= test__comparesf2(0x3f800001, 0x3f800002, RESULT_LT);
  status |= test__comparesf2(0x3f800001, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x3ffffffc, 0x3ffffffd, RESULT_LT);
  status |= test__comparesf2(0x3fffffff, 0x00000001, RESULT_GT);
  status |= test__comparesf2(0x3fffffff, 0x40000000, RESULT_LT);
  status |= test__comparesf2(0x40000000, 0x3f800000, RESULT_GT);
  status |= test__comparesf2(0x40000000, 0x3fffffff, RESULT_GT);
  status |= test__comparesf2(0x40000000, 0x40000000, RESULT_EQ);
  status |= test__comparesf2(0x40000000, 0x40000001, RESULT_LT);
  status |= test__comparesf2(0x40000000, 0xc0000000, RESULT_GT);
  status |= test__comparesf2(0x40000000, 0xc0000001, RESULT_GT);
  status |= test__comparesf2(0x40000000, 0xc0a00000, RESULT_GT);
  status |= test__comparesf2(0x40000001, 0x3f800001, RESULT_GT);
  status |= test__comparesf2(0x40000001, 0x40000002, RESULT_LT);
  status |= test__comparesf2(0x40000001, 0xc0000002, RESULT_GT);
  status |= test__comparesf2(0x40000002, 0x3f800001, RESULT_GT);
  status |= test__comparesf2(0x40000002, 0x3f800003, RESULT_GT);
  status |= test__comparesf2(0x40000004, 0x40000003, RESULT_GT);
  status |= test__comparesf2(0x40400000, 0x40400000, RESULT_EQ);
  status |= test__comparesf2(0x407fffff, 0x407ffffe, RESULT_GT);
  status |= test__comparesf2(0x407fffff, 0x40800002, RESULT_LT);
  status |= test__comparesf2(0x40800001, 0x407fffff, RESULT_GT);
  status |= test__comparesf2(0x40a00000, 0x00000000, RESULT_GT);
  status |= test__comparesf2(0x40a00000, 0x80000000, RESULT_GT);
  status |= test__comparesf2(0x40a00000, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x40a00000, 0xc0a00000, RESULT_GT);
  status |= test__comparesf2(0x7d800001, 0x7d7fffff, RESULT_GT);
  status |= test__comparesf2(0x7e7fffff, 0x7e7ffffe, RESULT_GT);
  status |= test__comparesf2(0x7e7fffff, 0x7e800002, RESULT_LT);
  status |= test__comparesf2(0x7e800000, 0x7e7fffff, RESULT_GT);
  status |= test__comparesf2(0x7e800000, 0x7e800000, RESULT_EQ);
  status |= test__comparesf2(0x7e800000, 0x7e800001, RESULT_LT);
  status |= test__comparesf2(0x7e800001, 0x7e800000, RESULT_GT);
  status |= test__comparesf2(0x7e800001, 0x7f000001, RESULT_LT);
  status |= test__comparesf2(0x7e800001, 0xfe800000, RESULT_GT);
  status |= test__comparesf2(0x7e800002, 0x7e000003, RESULT_GT);
  status |= test__comparesf2(0x7e800004, 0x7e800003, RESULT_GT);
  status |= test__comparesf2(0x7efffffe, 0x7efffffe, RESULT_EQ);
  status |= test__comparesf2(0x7efffffe, 0x7effffff, RESULT_LT);
  status |= test__comparesf2(0x7efffffe, 0xfeffffff, RESULT_GT);
  status |= test__comparesf2(0x7effffff, 0x3f800000, RESULT_GT);
  status |= test__comparesf2(0x7effffff, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0x7effffff, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x7effffff, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x7f000000, 0x3f800000, RESULT_GT);
  status |= test__comparesf2(0x7f000000, 0x7f000000, RESULT_EQ);
  status |= test__comparesf2(0x7f000000, 0x7f800000, RESULT_LT);
  status |= test__comparesf2(0x7f000000, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x7f000000, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x7f000000, 0xff800000, RESULT_GT);
  status |= test__comparesf2(0x7f000001, 0x7f000000, RESULT_GT);
  status |= test__comparesf2(0x7f000001, 0x7f000002, RESULT_LT);
  status |= test__comparesf2(0x7f000001, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x7f000002, 0x7e800001, RESULT_GT);
  status |= test__comparesf2(0x7f7ffffe, 0x3f800000, RESULT_GT);
  status |= test__comparesf2(0x7f7ffffe, 0x7f7fffff, RESULT_LT);
  status |= test__comparesf2(0x7f7ffffe, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x7f7ffffe, 0xff7fffff, RESULT_GT);
  status |= test__comparesf2(0x7f7fffff, 0x00000001, RESULT_GT);
  status |= test__comparesf2(0x7f7fffff, 0x3f800000, RESULT_GT);
  status |= test__comparesf2(0x7f7fffff, 0x7f7fffff, RESULT_EQ);
  status |= test__comparesf2(0x7f7fffff, 0x7fbed1eb, RESULT_UN);
  status |= test__comparesf2(0x7f7fffff, 0x7fe15ee3, RESULT_UN);
  status |= test__comparesf2(0x7f7fffff, 0x80000001, RESULT_GT);
  status |= test__comparesf2(0x7f7fffff, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x00000000, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x00000001, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x007fffff, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x7f000000, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x7f7fffff, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x7f800000, RESULT_EQ);
  status |= test__comparesf2(0x7f800000, 0x7f91a4da, RESULT_UN);
  status |= test__comparesf2(0x7f800000, 0x7fd44a09, RESULT_UN);
  status |= test__comparesf2(0x7f800000, 0x80000000, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x80000001, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0x807fffff, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0xff7fffff, RESULT_GT);
  status |= test__comparesf2(0x7f800000, 0xff800000, RESULT_GT);
  status |= test__comparesf2(0x7f86d066, 0x00000000, RESULT_UN);
  status |= test__comparesf2(0x7f85a878, 0x00000001, RESULT_UN);
  status |= test__comparesf2(0x7f8c0dca, 0x007fffff, RESULT_UN);
  status |= test__comparesf2(0x7f822725, 0x3f800000, RESULT_UN);
  status |= test__comparesf2(0x7f853870, 0x7f7fffff, RESULT_UN);
  status |= test__comparesf2(0x7fbefc9d, 0x7f800000, RESULT_UN);
  status |= test__comparesf2(0x7f9f84a9, 0x7f81461b, RESULT_UN);
  status |= test__comparesf2(0x7f9e2c1d, 0x7fe4a313, RESULT_UN);
  status |= test__comparesf2(0x7fb0e6d0, 0x80000000, RESULT_UN);
  status |= test__comparesf2(0x7fac9171, 0x80000001, RESULT_UN);
  status |= test__comparesf2(0x7f824ae6, 0x807fffff, RESULT_UN);
  status |= test__comparesf2(0x7fa8b9a0, 0xbf800000, RESULT_UN);
  status |= test__comparesf2(0x7f92a1cd, 0xff7fffff, RESULT_UN);
  status |= test__comparesf2(0x7fbe5d29, 0xff800000, RESULT_UN);
  status |= test__comparesf2(0x7fcc9a57, 0x00000000, RESULT_UN);
  status |= test__comparesf2(0x7fec9d71, 0x00000001, RESULT_UN);
  status |= test__comparesf2(0x7fd5db76, 0x007fffff, RESULT_UN);
  status |= test__comparesf2(0x7fd003d9, 0x3f800000, RESULT_UN);
  status |= test__comparesf2(0x7fca0684, 0x7f7fffff, RESULT_UN);
  status |= test__comparesf2(0x7fc46aa0, 0x7f800000, RESULT_UN);
  status |= test__comparesf2(0x7ff72b19, 0x7faee637, RESULT_UN);
  status |= test__comparesf2(0x7fe9e0c1, 0x7fcc2788, RESULT_UN);
  status |= test__comparesf2(0x7fc571ea, 0x80000000, RESULT_UN);
  status |= test__comparesf2(0x7fd81a54, 0x80000001, RESULT_UN);
  status |= test__comparesf2(0x7febdfaf, 0x807fffff, RESULT_UN);
  status |= test__comparesf2(0x7ffa1f94, 0xbf800000, RESULT_UN);
  status |= test__comparesf2(0x7ff38fa0, 0xff7fffff, RESULT_UN);
  status |= test__comparesf2(0x7fdf3502, 0xff800000, RESULT_UN);
  status |= test__comparesf2(0x80000000, 0x00000000, RESULT_EQ);
  status |= test__comparesf2(0x80000000, 0x00000001, RESULT_LT);
  status |= test__comparesf2(0x80000000, 0x007fffff, RESULT_LT);
  status |= test__comparesf2(0x80000000, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0x80000000, 0x7f800000, RESULT_LT);
  status |= test__comparesf2(0x80000000, 0x7fbdfb72, RESULT_UN);
  status |= test__comparesf2(0x80000000, 0x7fdd528e, RESULT_UN);
  status |= test__comparesf2(0x80000000, 0x80000001, RESULT_GT);
  status |= test__comparesf2(0x80000000, 0x807fffff, RESULT_GT);
  status |= test__comparesf2(0x80000000, 0x80800000, RESULT_GT);
  status |= test__comparesf2(0x80000000, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x80000000, 0xff800000, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0x00000001, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x3f7fffff, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x3ffffffe, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x3fffffff, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x7effffff, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x7f7ffffe, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x7f7fffff, RESULT_LT);
  status |= test__comparesf2(0x80000001, 0x7fac481a, RESULT_UN);
  status |= test__comparesf2(0x80000001, 0x7fcf111d, RESULT_UN);
  status |= test__comparesf2(0x80000001, 0x80000001, RESULT_EQ);
  status |= test__comparesf2(0x80000001, 0xbf7fffff, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0xbf800000, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0xbffffffe, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0xbfffffff, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0xfeffffff, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0xff7ffffe, RESULT_GT);
  status |= test__comparesf2(0x80000001, 0xff7fffff, RESULT_GT);
  status |= test__comparesf2(0x80000002, 0x80000001, RESULT_LT);
  status |= test__comparesf2(0x80000003, 0x40400000, RESULT_LT);
  status |= test__comparesf2(0x80000003, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0x80000003, 0x80000002, RESULT_LT);
  status |= test__comparesf2(0x80000003, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0x80000004, 0x80000004, RESULT_EQ);
  status |= test__comparesf2(0x807ffffd, 0x807ffffe, RESULT_GT);
  status |= test__comparesf2(0x807fffff, 0x00000000, RESULT_LT);
  status |= test__comparesf2(0x807fffff, 0x007fffff, RESULT_LT);
  status |= test__comparesf2(0x807fffff, 0x7f800000, RESULT_LT);
  status |= test__comparesf2(0x807fffff, 0x7faf07f6, RESULT_UN);
  status |= test__comparesf2(0x807fffff, 0x7fd18a54, RESULT_UN);
  status |= test__comparesf2(0x807fffff, 0x80000000, RESULT_LT);
  status |= test__comparesf2(0x807fffff, 0x807ffffe, RESULT_LT);
  status |= test__comparesf2(0x807fffff, 0x807fffff, RESULT_EQ);
  status |= test__comparesf2(0x807fffff, 0x80800000, RESULT_GT);
  status |= test__comparesf2(0x807fffff, 0xff800000, RESULT_GT);
  status |= test__comparesf2(0x80800000, 0x00000000, RESULT_LT);
  status |= test__comparesf2(0x80800000, 0x00800000, RESULT_LT);
  status |= test__comparesf2(0x80800001, 0x80800000, RESULT_LT);
  status |= test__comparesf2(0x80800001, 0x80800002, RESULT_GT);
  status |= test__comparesf2(0x80ffffff, 0x81000000, RESULT_GT);
  status |= test__comparesf2(0x80ffffff, 0x81000002, RESULT_GT);
  status |= test__comparesf2(0x80ffffff, 0x81000004, RESULT_GT);
  status |= test__comparesf2(0x81000000, 0x80ffffff, RESULT_LT);
  status |= test__comparesf2(0x81000001, 0x80800001, RESULT_LT);
  status |= test__comparesf2(0x81000001, 0x80ffffff, RESULT_LT);
  status |= test__comparesf2(0x81000002, 0x80800001, RESULT_LT);
  status |= test__comparesf2(0x817fffff, 0x81800000, RESULT_GT);
  status |= test__comparesf2(0x81800000, 0x817fffff, RESULT_LT);
  status |= test__comparesf2(0x81800001, 0x817fffff, RESULT_LT);
  status |= test__comparesf2(0x81800002, 0x81000003, RESULT_LT);
  status |= test__comparesf2(0xbf800000, 0x3f800003, RESULT_LT);
  status |= test__comparesf2(0xbf800000, 0x7fa66ee9, RESULT_UN);
  status |= test__comparesf2(0xbf800000, 0x7fe481ef, RESULT_UN);
  status |= test__comparesf2(0xbf800000, 0x80000000, RESULT_LT);
  status |= test__comparesf2(0xbf800000, 0xbf800003, RESULT_GT);
  status |= test__comparesf2(0xbf800001, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0xbf800001, 0xbf800000, RESULT_LT);
  status |= test__comparesf2(0xbf800001, 0xbf800002, RESULT_GT);
  status |= test__comparesf2(0xbffffffc, 0xbffffffd, RESULT_GT);
  status |= test__comparesf2(0xbfffffff, 0x00000001, RESULT_LT);
  status |= test__comparesf2(0xbfffffff, 0xc0000000, RESULT_GT);
  status |= test__comparesf2(0xc0000000, 0x40000001, RESULT_LT);
  status |= test__comparesf2(0xc0000000, 0xbfffffff, RESULT_LT);
  status |= test__comparesf2(0xc0000000, 0xc0000001, RESULT_GT);
  status |= test__comparesf2(0xc0000001, 0x40000002, RESULT_LT);
  status |= test__comparesf2(0xc0000001, 0xbf800001, RESULT_LT);
  status |= test__comparesf2(0xc0000001, 0xc0000002, RESULT_GT);
  status |= test__comparesf2(0xc0000002, 0xbf800001, RESULT_LT);
  status |= test__comparesf2(0xc0000002, 0xbf800003, RESULT_LT);
  status |= test__comparesf2(0xc0000004, 0xc0000003, RESULT_LT);
  status |= test__comparesf2(0xc0400000, 0x40400000, RESULT_LT);
  status |= test__comparesf2(0xc07fffff, 0xc07ffffe, RESULT_LT);
  status |= test__comparesf2(0xc07fffff, 0xc0800002, RESULT_GT);
  status |= test__comparesf2(0xc0800001, 0xc07fffff, RESULT_LT);
  status |= test__comparesf2(0xfd800001, 0xfd7fffff, RESULT_LT);
  status |= test__comparesf2(0xfe7fffff, 0xfe7ffffe, RESULT_LT);
  status |= test__comparesf2(0xfe7fffff, 0xfe800002, RESULT_GT);
  status |= test__comparesf2(0xfe800000, 0xfe7fffff, RESULT_LT);
  status |= test__comparesf2(0xfe800000, 0xfe800001, RESULT_GT);
  status |= test__comparesf2(0xfe800001, 0x7e800000, RESULT_LT);
  status |= test__comparesf2(0xfe800001, 0xfe800000, RESULT_LT);
  status |= test__comparesf2(0xfe800001, 0xff000001, RESULT_GT);
  status |= test__comparesf2(0xfe800002, 0xfe000003, RESULT_LT);
  status |= test__comparesf2(0xfe800004, 0xfe800003, RESULT_LT);
  status |= test__comparesf2(0xfefffffe, 0x7efffffe, RESULT_LT);
  status |= test__comparesf2(0xfefffffe, 0x7effffff, RESULT_LT);
  status |= test__comparesf2(0xfefffffe, 0xfefffffe, RESULT_EQ);
  status |= test__comparesf2(0xfefffffe, 0xfeffffff, RESULT_GT);
  status |= test__comparesf2(0xfeffffff, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0xfeffffff, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0xfeffffff, 0xbf800000, RESULT_LT);
  status |= test__comparesf2(0xfeffffff, 0xff000000, RESULT_GT);
  status |= test__comparesf2(0xff000000, 0x00000000, RESULT_LT);
  status |= test__comparesf2(0xff000000, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0xff000000, 0x7f800000, RESULT_LT);
  status |= test__comparesf2(0xff000000, 0x80000000, RESULT_LT);
  status |= test__comparesf2(0xff000000, 0xbf800000, RESULT_LT);
  status |= test__comparesf2(0xff000000, 0xff000000, RESULT_EQ);
  status |= test__comparesf2(0xff000000, 0xff800000, RESULT_GT);
  status |= test__comparesf2(0xff000001, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0xff000001, 0xff000000, RESULT_LT);
  status |= test__comparesf2(0xff000001, 0xff000002, RESULT_GT);
  status |= test__comparesf2(0xff000002, 0xfe800001, RESULT_LT);
  status |= test__comparesf2(0xff7ffffe, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0xff7ffffe, 0x7f7fffff, RESULT_LT);
  status |= test__comparesf2(0xff7ffffe, 0xbf800000, RESULT_LT);
  status |= test__comparesf2(0xff7ffffe, 0xff7fffff, RESULT_GT);
  status |= test__comparesf2(0xff7fffff, 0x00000001, RESULT_LT);
  status |= test__comparesf2(0xff7fffff, 0x3f800000, RESULT_LT);
  status |= test__comparesf2(0xff7fffff, 0x7f919cff, RESULT_UN);
  status |= test__comparesf2(0xff7fffff, 0x7fd729a7, RESULT_UN);
  status |= test__comparesf2(0xff7fffff, 0x80000001, RESULT_LT);
  status |= test__comparesf2(0xff7fffff, 0xbf800000, RESULT_LT);
  status |= test__comparesf2(0xff7fffff, 0xff7fffff, RESULT_EQ);
  status |= test__comparesf2(0xff800000, 0x00000000, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x00000001, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x007fffff, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x7f000000, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x7f7fffff, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x7f800000, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x7fafdbc1, RESULT_UN);
  status |= test__comparesf2(0xff800000, 0x7fec80fe, RESULT_UN);
  status |= test__comparesf2(0xff800000, 0x80000000, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x80000001, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0x807fffff, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0xff000000, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0xff7fffff, RESULT_LT);
  status |= test__comparesf2(0xff800000, 0xff800000, RESULT_EQ);

  return status;
}
