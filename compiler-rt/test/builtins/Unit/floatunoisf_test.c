// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunoisf
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI float __floatunoisf(ou_int a);

int test__floatunoisf(ou_int a, float expected) {
  float x = __floatunoisf(a);
  if (x != expected) {
    printf("error in __floatunoisf: got %f, expected %f\n", x, expected);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__floatunoisf((ou_int)0, 0.0f))
    return 1;
  if (test__floatunoisf((ou_int)1, 1.0f))
    return 1;
  if (test__floatunoisf((ou_int)42, 42.0f))
    return 1;
  if (test__floatunoisf((ou_int)100, 100.0f))
    return 1;
  if (test__floatunoisf((ou_int)1000000, 1e6f))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
