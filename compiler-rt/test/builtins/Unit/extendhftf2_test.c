// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendhftf2

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113 && defined(CRT_HAS_FLOAT16)

#include "fp_test.h"

COMPILER_RT_ABI long double __extendhftf2(TYPE_FP16 a);

int test__extendhftf2(TYPE_FP16 a, uint64_t expectedHi, uint64_t expectedLo) {
  long double x = __extendhftf2(a);
  int ret = compareResultLD(x, expectedHi, expectedLo);

  if (ret) {
    printf("error in test__extendhftf2(%#.4x) = %.20Lf, "
           "expected %.20Lf\n",
           toRep16(a), x,
           fromRep128(expectedHi, expectedLo));
  }
  return ret;
}

char assumption_1[sizeof(TYPE_FP16) * CHAR_BIT == 16] = {0};

#endif

int main() {
#if __LDBL_MANT_DIG__ == 113 && defined(CRT_HAS_FLOAT16)
  // qNaN
  if (test__extendhftf2(makeQNaN16(),
                        UINT64_C(0x7fff800000000000),
                        UINT64_C(0x0)))
    return 1;
  // NaN
  if (test__extendhftf2(makeNaN16(UINT16_C(0x0100)),
                        UINT64_C(0x7fff400000000000),
                        UINT64_C(0x0)))
    return 1;
  // inf
  if (test__extendhftf2(makeInf16(),
                        UINT64_C(0x7fff000000000000),
                        UINT64_C(0x0)))
    return 1;
  if (test__extendhftf2(-makeInf16(),
                        UINT64_C(0xffff000000000000),
                        UINT64_C(0x0)))
    return 1;
  // zero
  if (test__extendhftf2(fromRep16(0x0U),
                        UINT64_C(0x0), UINT64_C(0x0)))
    return 1;
  if (test__extendhftf2(fromRep16(0x8000U),
                        UINT64_C(0x8000000000000000),
                        UINT64_C(0x0)))
    return 1;
  // denormal
  if (test__extendhftf2(fromRep16(0x0010U),
                        UINT64_C(0x3feb000000000000),
                        UINT64_C(0x0000000000000000)))
    return 1;
  if (test__extendhftf2(fromRep16(0x0001U),
                        UINT64_C(0x3fe7000000000000),
                        UINT64_C(0x0000000000000000)))
    return 1;
  if (test__extendhftf2(fromRep16(0x8001U),
                        UINT64_C(0xbfe7000000000000),
                        UINT64_C(0x0000000000000000)))
    return 1;

  // pi
  if (test__extendhftf2(fromRep16(0x4248U),
                        UINT64_C(0x4000920000000000),
                        UINT64_C(0x0000000000000000)))
    return 1;
  if (test__extendhftf2(fromRep16(0xc248U),
                        UINT64_C(0xc000920000000000),
                        UINT64_C(0x0000000000000000)))
    return 1;

  if (test__extendhftf2(fromRep16(0x508cU),
                        UINT64_C(0x4004230000000000),
                        UINT64_C(0x0)))
    return 1;
  if (test__extendhftf2(fromRep16(0x1bb7U),
                        UINT64_C(0x3ff6edc000000000),
                        UINT64_C(0x0)))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
