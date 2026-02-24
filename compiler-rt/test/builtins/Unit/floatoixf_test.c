// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatoixf
// REQUIRES: x86-target-arch
// REQUIRES: int256

#include "int_lib.h"
#include <float.h>
#include <stdio.h>

#if defined(CRT_HAS_256BIT) && HAS_80_BIT_LONG_DOUBLE

// Returns: convert a to a long double, rounding toward even.

// Assumption: long double is a IEEE 80 bit floating point type padded to 128
// bits
//             oi_int is a 256 bit integral type

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee
// eeee | 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm
// mmmm mmmm mmmm

COMPILER_RT_ABI long double __floatoixf(oi_int a);

int test__floatoixf(oi_int a, long double expected) {
  long double x = __floatoixf(a);
  if (x != expected) {
    printf("error in __floatoixf = %LA, expected %LA\n", x, expected);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};
char assumption_2[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_256BIT) && HAS_80_BIT_LONG_DOUBLE
  if (test__floatoixf(0, 0.0))
    return 1;

  if (test__floatoixf(1, 1.0))
    return 1;
  if (test__floatoixf(2, 2.0))
    return 1;
  if (test__floatoixf(20, 20.0))
    return 1;
  if (test__floatoixf(-1, -1.0))
    return 1;
  if (test__floatoixf(-2, -2.0))
    return 1;
  if (test__floatoixf(-20, -20.0))
    return 1;

  // Precision boundary tests (from 128-bit reference)
  if (test__floatoixf(0x7FFFFF8000000000LL, 0x1.FFFFFEp+62))
    return 1;
  if (test__floatoixf(0x7FFFFFFFFFFFF800LL, 0x1.FFFFFFFFFFFFEp+62))
    return 1;
  if (test__floatoixf(0x7FFFFF0000000000LL, 0x1.FFFFFCp+62))
    return 1;
  if (test__floatoixf(0x7FFFFFFFFFFFF000LL, 0x1.FFFFFFFFFFFFCp+62))
    return 1;

  // Full long double precision (64-bit mantissa)
  if (test__floatoixf(0x7FFFFFFFFFFFFFFFLL, 0xF.FFFFFFFFFFFFFFEp+59L))
    return 1;
  if (test__floatoixf(0x023479FD0E092DC0LL, 0x1.1A3CFE870496Ep+57))
    return 1;
  if (test__floatoixf(0x023479FD0E092DA1LL, 0x1.1A3CFE870496D08p+57L))
    return 1;

  // Values spanning >64 bits (128-bit range, in oi_int)
  if (test__floatoixf(make_oi(make_ti(0, 0), make_ti(0x023479FD0E092DC0LL, 0)),
                      0x1.1A3CFE870496Ep+121L))
    return 1;

  // Negative values
  if (test__floatoixf(make_oi(make_ti(0xFFFFFFFFFFFFFFFFLL, -1),
                              make_ti(0x8000008000000000LL, 0)),
                      -0x1.FFFFFEp+126))
    return 1;
  if (test__floatoixf(make_oi(make_ti(0xFFFFFFFFFFFFFFFFLL, -1),
                              make_ti(0x8000000000000000LL, 0)),
                      -0x1.000000p+127))
    return 1;
  if (test__floatoixf(make_oi(make_ti(0xFFFFFFFFFFFFFFFFLL, -1),
                              make_ti(0x8000000000000001LL, 0)),
                      -0x1.FFFFFFFFFFFFFFFCp+126L))
    return 1;

  // Values beyond 128-bit range: high half set
  if (test__floatoixf(make_oi(make_ti(0, 1), make_ti(0, 0)), 0x1.0p+128L))
    return 1;
  // 2^200
  if (test__floatoixf((oi_int)1 << 200, 0x1.0p+200L))
    return 1;

  // Large 256-bit value near max
  if (test__floatoixf(make_oi(make_ti(0x023479FD0E092DC0LL, 0), make_ti(0, 0)),
                      0x1.1A3CFE870496Ep+249L))
    return 1;

  // Max unsigned 64-bit in lower half
  if (test__floatoixf(make_oi(make_ti(0, 0), make_ti(0, 0xFFFFFFFFFFFFFFFFLL)),
                      0x1.FFFFFFFFFFFFFFFEp+63L))
    return 1;

#else
  printf("skipped\n");
#endif
  return 0;
}
