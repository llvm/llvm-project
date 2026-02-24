// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunssfoi
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI ou_int __fixunssfoi(float a);

int test__fixunssfoi(float a, ou_int expected) {
  ou_int x = __fixunssfoi(a);
  if (x != expected) {
    printf("error in __fixunssfoi(%f)\n", a);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__fixunssfoi(0.0f, (ou_int)0))
    return 1;
  if (test__fixunssfoi(1.0f, (ou_int)1))
    return 1;
  if (test__fixunssfoi(42.0f, (ou_int)42))
    return 1;
  if (test__fixunssfoi(-1.0f, (ou_int)0))
    return 1;
  if (test__fixunssfoi(0.5f, (ou_int)0))
    return 1;
  if (test__fixunssfoi(1.5f, (ou_int)1))
    return 1;
  if (test__fixunssfoi(100.0f, (ou_int)100))
    return 1;
  if (test__fixunssfoi(1e6f, (ou_int)1000000))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
