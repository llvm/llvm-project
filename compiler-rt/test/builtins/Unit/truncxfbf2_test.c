// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_truncxfbf2

#include <stdio.h>
#include <string.h>

#include "fp_test.h"

#if HAS_80_BIT_LONG_DOUBLE && defined(__x86_64__)

__bf16 __truncxfbf2(xf_float f);

int test_truncxfbf2(uint16_t inputHi, uint64_t inputLo, uint16_t e) {
  xf_float a = F80FromRep80(inputHi, inputLo);
  __bf16 x = __truncxfbf2(a);
  uint16_t rep;
  memcpy(&rep, &x, sizeof(rep));
  if (rep != e) {
    printf("error in test__truncxfbf2(%Lf) = %#.4x, expected %#.4x\n", a, rep,
           e);
    return 1;
  }
  return 0;
}

int main() {
  // 1
  if (test_truncxfbf2(UINT16_C(0x3fff), UINT64_C(0x8000000000000000),
                      UINT16_C(0x3f80)))
    return 1;

  // -1.5
  if (test_truncxfbf2(UINT16_C(0xbfff), UINT64_C(0xc000000000000000),
                      UINT16_C(0xbfc0)))
    return 1;

  // Zero
  if (test_truncxfbf2(UINT16_C(0x0), UINT64_C(0x0), UINT16_C(0x0)))
    return 1;

  // 1 + 2^-8 + 2^-40, just above the midpoint between 1 and 1 + 2^-7
  if (test_truncxfbf2(UINT16_C(0x3fff), UINT64_C(0x8080000000800000),
                      UINT16_C(0x3f81)))
    return 1;

  // 1 + 2^-8, the midpoint, rounds to even
  if (test_truncxfbf2(UINT16_C(0x3fff), UINT64_C(0x8080000000000000),
                      UINT16_C(0x3f80)))
    return 1;

  // Positive infinity
  if (test_truncxfbf2(UINT16_C(0x7fff), UINT64_C(0x8000000000000000),
                      UINT16_C(0x7f80)))
    return 1;

  // Too large for bf16
  if (test_truncxfbf2(UINT16_C(0x43ff), UINT64_C(0x8000000000000000),
                      UINT16_C(0x7f80)))
    return 1;

  // NaN
  if (test_truncxfbf2(UINT16_C(0x7fff), UINT64_C(0xc000000000000000),
                      UINT16_C(0x7fc0)))
    return 1;

  return 0;
}

#else

int main() {
  printf("skipped\n");
  return 0;
}

#endif
