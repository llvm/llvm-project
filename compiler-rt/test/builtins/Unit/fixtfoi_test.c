// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixtfoi
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#if defined(CRT_HAS_256BIT) && __LDBL_MANT_DIG__ == 113

COMPILER_RT_ABI oi_int __fixtfoi(long double a);

int test__fixtfoi(long double a, oi_int expected) {
  oi_int x = __fixtfoi(a);
  if (x != expected) {
    printf("error in __fixtfoi\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#if defined(CRT_HAS_256BIT) && __LDBL_MANT_DIG__ == 113
  if (test__fixtfoi(0.0L, (oi_int)0))
    return 1;
  if (test__fixtfoi(1.0L, (oi_int)1))
    return 1;
  if (test__fixtfoi(-1.0L, (oi_int)-1))
    return 1;
  if (test__fixtfoi(42.0L, (oi_int)42))
    return 1;
  if (test__fixtfoi(-42.0L, (oi_int)-42))
    return 1;
  if (test__fixtfoi(0.5L, (oi_int)0))
    return 1;
  if (test__fixtfoi(1.5L, (oi_int)1))
    return 1;
  if (test__fixtfoi(-0.5L, (oi_int)0))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
