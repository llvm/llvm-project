// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendhfxf2

#include <limits.h>
#include <math.h> // for isnan, isinf
#include <stdio.h>

#include "fp_test.h"

#if HAS_80_BIT_LONG_DOUBLE

xf_float __extendhfxf2(TYPE_FP16 f);

int test_extendhfxf2(TYPE_FP16 a, uint16_t expectedHi, uint64_t expectedLo) {
  xf_float x = __extendhfxf2(a);
  int ret = compareResultF80(x, expectedHi, expectedLo);
  if (ret) {
    printf("error in test__extendhfxf2(%#.4x) = %.20Lf, "
           "expected %.20Lf\n",
           toRep16(a), x, F80FromRep80(expectedHi, expectedLo));
  }
  return ret;
}

int main() {
  // Small positive value
  if (test_extendhfxf2(fromRep16(0x2e66), UINT16_C(0x3ffb),
                       UINT64_C(0xccc0000000000000)))
    return 1;

  // Small negative value
  if (test_extendhfxf2(fromRep16(0xae66), UINT16_C(0xbffb),
                       UINT64_C(0xccc0000000000000)))
    return 1;

  // Zero
  if (test_extendhfxf2(fromRep16(0), UINT16_C(0x0), UINT64_C(0x0)))
    return 1;

  // Smallest positive non-zero value
  if (test_extendhfxf2(fromRep16(0x0100), UINT16_C(0x3fef),
                       UINT64_C(0x8000000000000000)))
    return 1;

  // Smallest negative non-zero value
  if (test_extendhfxf2(fromRep16(0x8100), UINT16_C(0xbfef),
                       UINT64_C(0x8000000000000000)))
    return 1;

  // Positive infinity
  if (test_extendhfxf2(makeInf16(), UINT16_C(0x7fff),
                       UINT64_C(0x8000000000000000)))
    return 1;

  // Negative infinity
  if (test_extendhfxf2(makeNegativeInf16(), UINT16_C(0xffff),
                       UINT64_C(0x8000000000000000)))
    return 1;

  // NaN
  if (test_extendhfxf2(makeQNaN16(), UINT16_C(0x7fff),
                       UINT64_C(0xc000000000000000)))
    return 1;

  return 0;
}

#else

int main() {
  printf("skipped\n");
  return 0;
}

#endif
