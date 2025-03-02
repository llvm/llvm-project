// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_truncxfhf2

#include <stdio.h>

#include "fp_test.h"

#if HAS_80_BIT_LONG_DOUBLE

TYPE_FP16 __truncxfhf2(xf_float f);

int test_truncxfhf2(uint16_t inputHi, uint64_t inputLo, uint16_t e) {
  xf_float a = F80FromRep80(inputHi, inputLo);
  TYPE_FP16 x = __truncxfhf2(a);
  int ret = compareResultH(x, e);
  if (ret) {
    printf("error in test__truncxfhf2(%Lf) = %#.4x, "
           "expected %#.4x\n",
           a, toRep16(x), e);
  }
  return ret;
}

int main() {
  // Small positive value
  if (test_truncxfhf2(UINT16_C(0x3ffb), UINT64_C(0xccc0000000000000),
                      UINT16_C(0x2e66)))
    return 1;

  // Small negative value
  if (test_truncxfhf2(UINT16_C(0xbffb), UINT64_C(0xccc0000000000000),
                      UINT16_C(0xae66)))
    return 1;

  // Zero
  if (test_truncxfhf2(UINT16_C(0x0), UINT64_C(0x0), UINT16_C(0)))
    return 1;

  // Smallest positive non-zero value
  if (test_truncxfhf2(UINT16_C(0x3fef), UINT64_C(0x8000000000000000),
                      UINT16_C(0x0100)))
    return 1;

  // Smallest negative non-zero value
  if (test_truncxfhf2(UINT16_C(0xbfef), UINT64_C(0x8000000000000000),
                      UINT16_C(0x8100)))
    return 1;

  // Positive infinity
  if (test_truncxfhf2(UINT16_C(0x7fff), UINT64_C(0x8000000000000000),
                      UINT16_C(0x7c00)))
    return 1;

  // Negative infinity
  if (test_truncxfhf2(UINT16_C(0xffff), UINT64_C(0x8000000000000000),
                      UINT16_C(0xfc00)))
    return 1;

  // NaN
  if (test_truncxfhf2(UINT16_C(0x7fff), UINT64_C(0xc000000000000000),
                      UINT16_C(0x7e00)))
    return 1;

  return 0;
}

#else

int main() {
  printf("skipped\n");
  return 0;
}

#endif
