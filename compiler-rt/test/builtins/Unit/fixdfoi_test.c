// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixdfoi
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI oi_int __fixdfoi(double a);

int test__fixdfoi(double a, oi_int expected) {
  oi_int x = __fixdfoi(a);
  if (x != expected) {
    printf("error in __fixdfoi(%f)\n", a);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__fixdfoi(0.0, (oi_int)0))
    return 1;
  if (test__fixdfoi(1.0, (oi_int)1))
    return 1;
  if (test__fixdfoi(-1.0, (oi_int)-1))
    return 1;
  if (test__fixdfoi(42.0, (oi_int)42))
    return 1;
  if (test__fixdfoi(-42.0, (oi_int)-42))
    return 1;
  if (test__fixdfoi(1e18, (oi_int)1000000000000000000LL))
    return 1;
  if (test__fixdfoi(0.5, (oi_int)0))
    return 1;
  if (test__fixdfoi(-0.5, (oi_int)0))
    return 1;
  if (test__fixdfoi(1.5, (oi_int)1))
    return 1;
  if (test__fixdfoi(-1.5, (oi_int)-1))
    return 1;
  if (test__fixdfoi(100.0, (oi_int)100))
    return 1;
  if (test__fixdfoi(-100.0, (oi_int)-100))
    return 1;
  // Rounding toward zero
  if (test__fixdfoi(0.99, (oi_int)0))
    return 1;
  if (test__fixdfoi(1.99, (oi_int)1))
    return 1;
  if (test__fixdfoi(-0.99, (oi_int)0))
    return 1;
  if (test__fixdfoi(-1.99, (oi_int)-1))
    return 1;
  if (test__fixdfoi(2.01, (oi_int)2))
    return 1;
  // Double mantissa boundary: 52 bits (53 with implicit 1)
  // 0x1.FFFFFFFFFFFFFp+62 = max double < 2^63
  if (test__fixdfoi(0x1.FFFFFFFFFFFFFp+62, (oi_int)0x7FFFFFFFFFFFFC00LL))
    return 1;
  if (test__fixdfoi(-0x1.FFFFFFFFFFFFFp+62, -(oi_int)0x7FFFFFFFFFFFFC00LL))
    return 1;
  // Exact powers of 2 in the 128+ bit range
  if (test__fixdfoi(0x1.0p+64, (oi_int)1 << 64))
    return 1;
  if (test__fixdfoi(0x1.0p+127, (oi_int)1 << 127))
    return 1;
  if (test__fixdfoi(0x1.0p+200, (oi_int)1 << 200))
    return 1;
  // Negative large
  if (test__fixdfoi(-0x1.0p+127, -((oi_int)1 << 127)))
    return 1;
  // Values at the double mantissa limit (52-bit precision):
  // 0x1.FFFFFFFFFFFFFp+126 -- max double in ~127-bit range
  if (test__fixdfoi(0x1.FFFFFFFFFFFFFp+126,
                    make_oi(make_ti(0, 0), make_ti(0x7FFFFFFFFFFFFC00LL, 0))))
    return 1;
  if (test__fixdfoi(-0x1.FFFFFFFFFFFFFp+126,
                    make_oi(make_ti(-1, -1), make_ti(0x8000000000000400LL, 0))))
    return 1;
  // Specific hex value (from 128-bit reference test)
  if (test__fixdfoi(0x1.1A3CFE870496Ep+57, (oi_int)0x023479FD0E092DC0LL))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
