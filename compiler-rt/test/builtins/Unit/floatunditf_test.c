// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunditf

#include "int_lib.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>

// The testcase currently assumes IEEE TF format, once that has been
// fixed the defined(CRT_HAS_IEEE_TF) guard can be removed to enable it for
// IBM 128 floats as well.
#if defined(CRT_HAS_IEEE_TF)

#  include "fp_test.h"

// Returns: long integer converted to long double

COMPILER_RT_ABI tf_float __floatunditf(du_int a);

int test__floatunditf(du_int a, uint64_t expectedHi, uint64_t expectedLo) {
  tf_float x = __floatunditf(a);
  int ret = compareResultF128(x, expectedHi, expectedLo);

  if (ret)
    printf("error in __floatunditf(%Lu) = %.20Lf, "
           "expected %.20Lf\n",
           a, x, fromRep128(expectedHi, expectedLo));
  return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_IEEE_TF)
  if (test__floatunditf(0xffffffffffffffffULL, UINT64_C(0x403effffffffffff),
                        UINT64_C(0xfffe000000000000)))
    return 1;
  if (test__floatunditf(0xfffffffffffffffeULL, UINT64_C(0x403effffffffffff),
                        UINT64_C(0xfffc000000000000)))
    return 1;
  if (test__floatunditf(0x8000000000000000ULL, UINT64_C(0x403e000000000000),
                        UINT64_C(0x0)))
    return 1;
  if (test__floatunditf(0x7fffffffffffffffULL, UINT64_C(0x403dffffffffffff),
                        UINT64_C(0xfffc000000000000)))
    return 1;
  if (test__floatunditf(0x123456789abcdef1ULL, UINT64_C(0x403b23456789abcd),
                        UINT64_C(0xef10000000000000)))
    return 1;
  if (test__floatunditf(0x2ULL, UINT64_C(0x4000000000000000), UINT64_C(0x0)))
    return 1;
  if (test__floatunditf(0x1ULL, UINT64_C(0x3fff000000000000), UINT64_C(0x0)))
    return 1;
  if (test__floatunditf(0x0ULL, UINT64_C(0x0), UINT64_C(0x0)))
    return 1;

#else
  printf("skipped\n");

#endif
  return 0;
}
