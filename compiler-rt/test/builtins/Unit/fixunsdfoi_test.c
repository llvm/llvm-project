// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunsdfoi
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI ou_int __fixunsdfoi(double a);

int test__fixunsdfoi(double a, ou_int expected) {
  ou_int x = __fixunsdfoi(a);
  if (x != expected) {
    printf("error in __fixunsdfoi(%f)\n", a);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__fixunsdfoi(0.0, (ou_int)0))
    return 1;
  if (test__fixunsdfoi(1.0, (ou_int)1))
    return 1;
  if (test__fixunsdfoi(42.0, (ou_int)42))
    return 1;
  if (test__fixunsdfoi(1e18, (ou_int)1000000000000000000ULL))
    return 1;
  if (test__fixunsdfoi(-1.0, (ou_int)0))
    return 1;
  if (test__fixunsdfoi(0.5, (ou_int)0))
    return 1;
  if (test__fixunsdfoi(1.5, (ou_int)1))
    return 1;
  if (test__fixunsdfoi(100.0, (ou_int)100))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
