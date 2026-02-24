// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunoidf
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI double __floatunoidf(ou_int a);

int test__floatunoidf(ou_int a, double expected) {
  double x = __floatunoidf(a);
  if (x != expected) {
    printf("error in __floatunoidf: got %f, expected %f\n", x, expected);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__floatunoidf((ou_int)0, 0.0))
    return 1;
  if (test__floatunoidf((ou_int)1, 1.0))
    return 1;
  if (test__floatunoidf((ou_int)42, 42.0))
    return 1;
  if (test__floatunoidf((ou_int)1000000, 1e6))
    return 1;
  if (test__floatunoidf((ou_int)1000000000000000000ULL, 1e18))
    return 1;
  if (test__floatunoidf((ou_int)100, 100.0))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
