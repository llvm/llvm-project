// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_trunctfhf2

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113 && defined(CRT_HAS_FLOAT16)

#include "fp_test.h"

TYPE_FP16 __trunctfhf2(long double a);

int test__trunctfhf2(long double a, uint16_t expected) {
  TYPE_FP16 x = __trunctfhf2(a);
  int ret = compareResultH(x, expected);

  if (ret) {
    printf("error in test__trunctfhf2(%.20Lf) = %#.4x, "
           "expected %#.4x\n",
           a, toRep16(x), expected);
  }
  return ret;
}

char assumption_1[sizeof(TYPE_FP16) * CHAR_BIT == 16] = {0};

#endif

int main() {
#if __LDBL_MANT_DIG__ == 113 && defined(CRT_HAS_FLOAT16)
  // qNaN
  if (test__trunctfhf2(makeQNaN128(),
                       UINT16_C(0x7e00)))
    return 1;
  // NaN
  if (test__trunctfhf2(makeNaN128(UINT64_C(0x810000000000)),
                       UINT16_C(0x7e00)))
    return 1;
  // inf
  if (test__trunctfhf2(makeInf128(),
                       UINT16_C(0x7c00)))
    return 1;
  if (test__trunctfhf2(-makeInf128(),
                       UINT16_C(0xfc00)))
    return 1;
  // zero
  if (test__trunctfhf2(0.0L, UINT16_C(0x0)))
    return 1;
  if (test__trunctfhf2(-0.0L, UINT16_C(0x8000)))
    return 1;

  if (test__trunctfhf2(3.1415926535L,
                       UINT16_C(0x4248)))
    return 1;
  if (test__trunctfhf2(-3.1415926535L,
                       UINT16_C(0xc248)))
    return 1;
  if (test__trunctfhf2(0x1.987124876876324p+100L,
                       UINT16_C(0x7c00)))
    return 1;
  if (test__trunctfhf2(0x1.987124876876324p+12L,
                       UINT16_C(0x6e62)))
    return 1;
  if (test__trunctfhf2(0x1.0p+0L,
                       UINT16_C(0x3c00)))
    return 1;
  if (test__trunctfhf2(0x1.0p-14L,
                       UINT16_C(0x0400)))
    return 1;
  // denormal
  if (test__trunctfhf2(0x1.0p-20L,
                       UINT16_C(0x0010)))
    return 1;
  if (test__trunctfhf2(0x1.0p-24L,
                       UINT16_C(0x0001)))
    return 1;
  if (test__trunctfhf2(-0x1.0p-24L,
                       UINT16_C(0x8001)))
    return 1;
  if (test__trunctfhf2(0x1.5p-25L,
                       UINT16_C(0x0001)))
    return 1;
  // and back to zero
  if (test__trunctfhf2(0x1.0p-25L,
                       UINT16_C(0x0000)))
    return 1;
  if (test__trunctfhf2(-0x1.0p-25L,
                       UINT16_C(0x8000)))
    return 1;
  // max (precise)
  if (test__trunctfhf2(65504.0L,
                       UINT16_C(0x7bff)))
    return 1;
  // max (rounded)
  if (test__trunctfhf2(65519.0L,
                       UINT16_C(0x7bff)))
    return 1;
  // max (to +inf)
  if (test__trunctfhf2(65520.0L,
                       UINT16_C(0x7c00)))
    return 1;
  if (test__trunctfhf2(65536.0L,
                       UINT16_C(0x7c00)))
    return 1;
  if (test__trunctfhf2(-65520.0L,
                       UINT16_C(0xfc00)))
    return 1;

  if (test__trunctfhf2(0x1.23a2abb4a2ddee355f36789abcdep+5L,
                       UINT16_C(0x508f)))
    return 1;
  if (test__trunctfhf2(0x1.e3d3c45bd3abfd98b76a54cc321fp-9L,
                       UINT16_C(0x1b8f)))
    return 1;
  if (test__trunctfhf2(0x1.234eebb5faa678f4488693abcdefp+453L,
                       UINT16_C(0x7c00)))
    return 1;
  if (test__trunctfhf2(0x1.edcba9bb8c76a5a43dd21f334634p-43L,
                       UINT16_C(0x0)))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
