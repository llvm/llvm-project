// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatoidf
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI double __floatoidf(oi_int a);

int test__floatoidf(oi_int a, double expected) {
  double x = __floatoidf(a);
  if (x != expected) {
    printf("error in __floatoidf: got %f, expected %f\n", x, expected);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__floatoidf((oi_int)0, 0.0))
    return 1;
  if (test__floatoidf((oi_int)1, 1.0))
    return 1;
  if (test__floatoidf((oi_int)-1, -1.0))
    return 1;
  if (test__floatoidf((oi_int)42, 42.0))
    return 1;
  if (test__floatoidf((oi_int)-42, -42.0))
    return 1;
  if (test__floatoidf((oi_int)1000000, 1e6))
    return 1;
  if (test__floatoidf((oi_int)-1000000, -1e6))
    return 1;
  if (test__floatoidf((oi_int)100, 100.0))
    return 1;
  if (test__floatoidf((oi_int)20, 20.0))
    return 1;
  if (test__floatoidf((oi_int)-20, -20.0))
    return 1;
  // Double mantissa boundary: 52 bits (53 with implicit 1)
  // 2^53 = 9007199254740992, exactly representable
  if (test__floatoidf((oi_int)9007199254740992LL, 9007199254740992.0))
    return 1;
  // 2^53 + 1: NOT exactly representable, rounds to 2^53
  if (test__floatoidf((oi_int)9007199254740993LL, 9007199254740992.0))
    return 1;
  // 2^53 + 2: exactly representable
  if (test__floatoidf((oi_int)9007199254740994LL, 9007199254740994.0))
    return 1;
  // Specific values from 128-bit reference tests
  if (test__floatoidf((oi_int)0x7FFFFF8000000000LL, 0x1.FFFFFEp+62))
    return 1;
  if (test__floatoidf((oi_int)0x7FFFFFFFFFFFF800LL, 0x1.FFFFFFFFFFFFEp+62))
    return 1;
  // Large values spanning >64 bits
  if (test__floatoidf((oi_int)1 << 64, 0x1.0p+64))
    return 1;
  if (test__floatoidf((oi_int)1 << 127, 0x1.0p+127))
    return 1;
  if (test__floatoidf(-((oi_int)1 << 127), -0x1.0p+127))
    return 1;
  // Very large value: 2^200
  if (test__floatoidf((oi_int)1 << 200, 0x1.0p+200))
    return 1;
  // Values with high-half mantissa bits:
  // make_oi(make_ti(0x7FFFFF8000000000, 0), make_ti(0, 0))
  // = 0x7FFFFF8000000000 << 128, leading 1 at bit 254
  if (test__floatoidf(make_oi(make_ti(0x7FFFFF8000000000LL, 0), make_ti(0, 0)),
                      0x1.FFFFFEp+254))
    return 1;
  // Negative large
  if (test__floatoidf(make_oi(make_ti(0x8000008000000000LL, 0), make_ti(0, 0)),
                      -0x1.FFFFFEp+254))
    return 1;
  // Specific hex value (adapted from 128-bit reference)
  if (test__floatoidf((oi_int)0x023479FD0E092DC0LL, 0x1.1A3CFE870496Ep+57))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
