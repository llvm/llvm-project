// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendhfxf2

#include <limits.h>
#include <math.h> // for isnan, isinf
#include <stdio.h>

#include "int_lib.h"

#if HAS_80_BIT_LONG_DOUBLE && defined(COMPILER_RT_HAS_FLOAT16)

long double __extendhfxf2(_Float16 f);

int test_extendhfxf2(_Float16 a, long double expected) {
  long double x = __extendhfxf2(a);
  __uint16_t *b = (void *)&a;
  int ret = !((isnan(x) && isnan(expected)) || x == expected);
  if (ret) {
    printf("error in test__extendhfxf2(%#.4x) = %.20Lf, "
           "expected %.20Lf\n",
           *b, x, expected);
  }
  return ret;
}

char assumption_1[sizeof(_Float16) * CHAR_BIT == 16] = {0};

int main() {
  // Small positive value
  if (test_extendhfxf2(0.09997558593750000000f, 0.09997558593750000000L))
    return 1;

  // Small negative value
  if (test_extendhfxf2(-0.09997558593750000000f, -0.09997558593750000000L))
    return 1;

  // Zero
  if (test_extendhfxf2(0.0f, 0.0L))
    return 1;

  // Smallest positive non-zero value
  if (test_extendhfxf2(0x1p-16f, 0x1p-16L))
    return 1;

  // Smallest negative non-zero value
  if (test_extendhfxf2(-0x1p-16f, -0x1p-16L))
    return 1;

  // Positive infinity
  if (test_extendhfxf2(__builtin_huge_valf16(), __builtin_huge_valf64x()))
    return 1;

  // Negative infinity
  if (test_extendhfxf2(-__builtin_huge_valf16(),
                       (long double)-__builtin_huge_valf64x()))
    return 1;

  // NaN
  if (test_extendhfxf2(__builtin_nanf16(""),
                       (long double)__builtin_nanf64x("")))
    return 1;

  return 0;
}

#else

int main() {
  printf("skipped\n");
  return 0;
}

#endif
