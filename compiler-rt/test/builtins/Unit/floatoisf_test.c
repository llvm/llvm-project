// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatoisf
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI float __floatoisf(oi_int a);

int test__floatoisf(oi_int a, float expected) {
  float x = __floatoisf(a);
  if (x != expected) {
    printf("error in __floatoisf: got %f, expected %f\n", x, expected);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__floatoisf((oi_int)0, 0.0f))
    return 1;
  if (test__floatoisf((oi_int)1, 1.0f))
    return 1;
  if (test__floatoisf((oi_int)-1, -1.0f))
    return 1;
  if (test__floatoisf((oi_int)42, 42.0f))
    return 1;
  if (test__floatoisf((oi_int)-42, -42.0f))
    return 1;
  if (test__floatoisf((oi_int)100, 100.0f))
    return 1;
  if (test__floatoisf((oi_int)-100, -100.0f))
    return 1;
  if (test__floatoisf((oi_int)1000000, 1e6f))
    return 1;
  if (test__floatoisf((oi_int)-1000000, -1e6f))
    return 1;
  if (test__floatoisf((oi_int)20, 20.0f))
    return 1;
  if (test__floatoisf((oi_int)-20, -20.0f))
    return 1;
  // Precision boundary: float has 23 mantissa bits (24 with implicit 1)
  // 2^24 = 16777216, exactly representable
  if (test__floatoisf((oi_int)16777216, 16777216.0f))
    return 1;
  // 2^24 + 1 = 16777217: NOT exactly representable in float,
  // rounds to 16777216.0f
  if (test__floatoisf((oi_int)16777217, 16777216.0f))
    return 1;
  // 2^24 + 2 = 16777218: exactly representable (even, rounds-to-even)
  if (test__floatoisf((oi_int)16777218, 16777218.0f))
    return 1;
  // Values at the mantissa boundary:
  // 0x7FFFFF8000000000 = mantissa all-ones shifted to bit 62
  if (test__floatoisf((oi_int)0x7FFFFF8000000000LL, 0x1.FFFFFEp+62F))
    return 1;
  // Large 256-bit value: 2^127
  if (test__floatoisf((oi_int)1 << 127, 0x1.0p+127F))
    return 1;
  // Large negative
  if (test__floatoisf(-((oi_int)1 << 127), -0x1.0p+127F))
    return 1;
  // Value > 128 bits: 2^200 exceeds float range, should return +inf
  if (test__floatoisf((oi_int)1 << 200, __builtin_inff()))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
