// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixsfoi
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __fixsfoi(float a);

int test__fixsfoi(float a, oi_int expected) {
  oi_int x = __fixsfoi(a);
  if (x != expected) {
    printf("error in __fixsfoi(%f)\n", a);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__fixsfoi(0.0f, (oi_int)0))
    return 1;
  if (test__fixsfoi(1.0f, (oi_int)1))
    return 1;
  if (test__fixsfoi(-1.0f, (oi_int)-1))
    return 1;
  if (test__fixsfoi(42.0f, (oi_int)42))
    return 1;
  if (test__fixsfoi(-42.0f, (oi_int)-42))
    return 1;
  if (test__fixsfoi(0.5f, (oi_int)0))
    return 1;
  if (test__fixsfoi(1.5f, (oi_int)1))
    return 1;
  if (test__fixsfoi(-0.5f, (oi_int)0))
    return 1;
  if (test__fixsfoi(-1.5f, (oi_int)-1))
    return 1;
  if (test__fixsfoi(100.0f, (oi_int)100))
    return 1;
  if (test__fixsfoi(-100.0f, (oi_int)-100))
    return 1;
  if (test__fixsfoi(1e6f, (oi_int)1000000))
    return 1;
  // Rounding toward zero for fractional parts
  if (test__fixsfoi(0.99f, (oi_int)0))
    return 1;
  if (test__fixsfoi(1.99f, (oi_int)1))
    return 1;
  if (test__fixsfoi(-0.99f, (oi_int)0))
    return 1;
  if (test__fixsfoi(-1.99f, (oi_int)-1))
    return 1;
  if (test__fixsfoi(2.0f, (oi_int)2))
    return 1;
  if (test__fixsfoi(2.01f, (oi_int)2))
    return 1;
  if (test__fixsfoi(-2.0f, (oi_int)-2))
    return 1;
  // Precision boundary: float has 23 mantissa bits
  // 0x1.FFFFFEp+62 = max float < 2^63, mantissa fully used
  if (test__fixsfoi(0x1.FFFFFEp+62F, (oi_int)0x7FFFFF8000000000LL))
    return 1;
  if (test__fixsfoi(-0x1.FFFFFEp+62F, -(oi_int)0x7FFFFF8000000000LL))
    return 1;
  // Large float that needs >64 bits to represent
  // 0x1.0p+64 = 2^64 = 18446744073709551616
  if (test__fixsfoi(0x1.0p+64F, (oi_int)1 << 64))
    return 1;
  // 0x1.0p+127 = 2^127
  if (test__fixsfoi(0x1.0p+127F, (oi_int)1 << 127))
    return 1;
  // Largest finite float: 0x1.FFFFFEp+127 = (2^24 - 1) * 2^104
  // This fits in oi_int (it's only ~128 bits).
  if (test__fixsfoi(0x1.FFFFFEp+127F, (oi_int)0xFFFFFF << 104))
    return 1;
  // Negative large
  if (test__fixsfoi(-0x1.0p+127F, -((oi_int)1 << 127)))
    return 1;
  // Infinity should saturate to max
  if (test__fixsfoi(__builtin_inff(), make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, -1),
                                              make_ti(-1, -1))))
    return 1;
  // Negative infinity should saturate to min
  if (test__fixsfoi(-__builtin_inff(),
                    make_oi(make_ti(0x8000000000000000LL, 0), make_ti(0, 0))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
