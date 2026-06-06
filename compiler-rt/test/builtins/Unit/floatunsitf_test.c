// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunsitf

#include "int_lib.h"
#include <stdio.h>

// The testcase currently assumes IEEE TF format, once that has been
// fixed the defined(CRT_HAS_IEEE_TF) guard can be removed to enable it for
// IBM 128 floats as well.
#if defined(CRT_HAS_IEEE_TF)

#  include "fp_test.h"

COMPILER_RT_ABI tf_float __floatunsitf(su_int a);

int test__floatunsitf(su_int a, uint64_t expectedHi, uint64_t expectedLo) {
  tf_float x = __floatunsitf(a);
  int ret = compareResultF128(x, expectedHi, expectedLo);

  if (ret) {
    printf("error in test__floatunsitf(%u) = %.20Lf, "
           "expected %.20Lf\n",
           a, x, fromRep128(expectedHi, expectedLo));
  }
  return ret;
}

char assumption_1[sizeof(tf_float) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_IEEE_TF)
  if (test__floatunsitf(0x7fffffff, UINT64_C(0x401dfffffffc0000),
                        UINT64_C(0x0)))
    return 1;
  if (test__floatunsitf(0, UINT64_C(0x0), UINT64_C(0x0)))
    return 1;
  if (test__floatunsitf(0xffffffff, UINT64_C(0x401efffffffe0000),
                        UINT64_C(0x0)))
    return 1;
  if (test__floatunsitf(0x12345678, UINT64_C(0x401b234567800000),
                        UINT64_C(0x0)))
    return 1;

#else
  printf("skipped\n");

#endif
  return 0;
}
