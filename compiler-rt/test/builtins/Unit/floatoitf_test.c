// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatoitf
// REQUIRES: int256

#define QUAD_PRECISION
#include "fp_lib.h"
#include "int_lib.h"
#include <stdio.h>

#if defined(CRT_HAS_TF_MODE) && defined(CRT_HAS_256BIT)

COMPILER_RT_ABI fp_t __floatoitf(oi_int a);

int test__floatoitf(oi_int a, fp_t expected) {
  fp_t x = __floatoitf(a);
  if (x != expected) {
    printf("error in __floatoitf\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#if defined(CRT_HAS_TF_MODE) && defined(CRT_HAS_256BIT)
  if (test__floatoitf((oi_int)0, TF_C(0.0)))
    return 1;
  if (test__floatoitf((oi_int)1, TF_C(1.0)))
    return 1;
  if (test__floatoitf((oi_int)-1, TF_C(-1.0)))
    return 1;
  if (test__floatoitf((oi_int)42, TF_C(42.0)))
    return 1;
  if (test__floatoitf((oi_int)-42, TF_C(-42.0)))
    return 1;
  if (test__floatoitf((oi_int)1000000, TF_C(1e6)))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
