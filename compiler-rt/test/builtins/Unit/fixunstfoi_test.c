// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunstfoi
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#if defined(CRT_HAS_256BIT) && __LDBL_MANT_DIG__ == 113

COMPILER_RT_ABI ou_int __fixunstfoi(long double a);

int test__fixunstfoi(long double a, ou_int expected) {
  ou_int x = __fixunstfoi(a);
  if (x != expected) {
    printf("error in __fixunstfoi\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#if defined(CRT_HAS_256BIT) && __LDBL_MANT_DIG__ == 113
  if (test__fixunstfoi(0.0L, (ou_int)0))
    return 1;
  if (test__fixunstfoi(1.0L, (ou_int)1))
    return 1;
  if (test__fixunstfoi(42.0L, (ou_int)42))
    return 1;
  if (test__fixunstfoi(0.5L, (ou_int)0))
    return 1;
  if (test__fixunstfoi(1.5L, (ou_int)1))
    return 1;
  if (test__fixunstfoi(1000000.0L, (ou_int)1000000))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
