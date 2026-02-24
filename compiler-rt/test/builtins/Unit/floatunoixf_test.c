// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunoixf
// REQUIRES: x86-target-arch
// REQUIRES: int256

#include "int_lib.h"
#include <float.h>
#include <stdio.h>

#if defined(CRT_HAS_256BIT) && HAS_80_BIT_LONG_DOUBLE

// Returns: convert a to a long double, rounding toward even.

// Assumption: long double is a IEEE 80 bit floating point type padded to 128
// bits
//             ou_int is a 256 bit integral type

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee
// eeee | 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm
// mmmm mmmm mmmm

COMPILER_RT_ABI long double __floatunoixf(ou_int a);

int test__floatunoixf(ou_int a, long double expected) {
  long double x = __floatunoixf(a);
  if (x != expected) {
    printf("error in __floatunoixf = %LA, expected %LA\n", x, expected);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(ou_int) == 2 * sizeof(tu_int)] = {0};
char assumption_2[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_256BIT) && HAS_80_BIT_LONG_DOUBLE
  if (test__floatunoixf(0, 0.0))
    return 1;

  if (test__floatunoixf(1, 1.0))
    return 1;
  if (test__floatunoixf(2, 2.0))
    return 1;
  if (test__floatunoixf(20, 20.0))
    return 1;

  // Precision boundary tests
  if (test__floatunoixf(0x7FFFFF8000000000ULL, 0x1.FFFFFEp+62))
    return 1;
  if (test__floatunoixf(0x7FFFFFFFFFFFF800ULL, 0x1.FFFFFFFFFFFFEp+62))
    return 1;
  if (test__floatunoixf(0x7FFFFF0000000000ULL, 0x1.FFFFFCp+62))
    return 1;
  if (test__floatunoixf(0x7FFFFFFFFFFFF000ULL, 0x1.FFFFFFFFFFFFCp+62))
    return 1;
  if (test__floatunoixf(0x7FFFFFFFFFFFFFFFULL, 0xF.FFFFFFFFFFFFFFEp+59L))
    return 1;
  if (test__floatunoixf(0xFFFFFFFFFFFFFFFEULL, 0xF.FFFFFFFFFFFFFFEp+60L))
    return 1;
  if (test__floatunoixf(0xFFFFFFFFFFFFFFFFULL, 0xF.FFFFFFFFFFFFFFFp+60L))
    return 1;

  // Specific hex value tests
  if (test__floatunoixf(0x8000008000000000ULL, 0x8.000008p+60))
    return 1;
  if (test__floatunoixf(0x8000000000000800ULL, 0x8.0000000000008p+60))
    return 1;
  if (test__floatunoixf(0x8000000000000000ULL, 0x8p+60))
    return 1;
  if (test__floatunoixf(0x8000000000000001ULL, 0x8.000000000000001p+60L))
    return 1;

  if (test__floatunoixf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50))
    return 1;
  if (test__floatunoixf(0x023479FD0E092DC0LL, 0x1.1A3CFE870496Ep+57))
    return 1;
  if (test__floatunoixf(0x023479FD0E092DA1LL, 0x1.1A3CFE870496D08p+57L))
    return 1;

  // Values spanning >64 bits (128-bit range, in ou_int)
  if (test__floatunoixf(
          make_oi(make_ti(0, 0), make_ti(0x023479FD0E092DC0LL, 0)),
          0x1.1A3CFE870496Ep+121L))
    return 1;

  // Max unsigned 128-bit value in lower half
  if (test__floatunoixf(make_oi(make_ti(0, 0), make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                       0xFFFFFFFFFFFFFFFFLL)),
                        0x1.0000000000000000p+128L))
    return 1;
  if (test__floatunoixf(
          make_oi(make_ti(0, 0), make_ti(0xFFFFFFFFFFFFFFFFLL, 0)),
          0x1.FFFFFFFFFFFFFFFEp+127L))
    return 1;

  // Values beyond 128-bit range: high half set
  if (test__floatunoixf(make_oi(make_ti(0, 1), make_ti(0, 0)), 0x1.0p+128L))
    return 1;
  // 2^200
  if (test__floatunoixf((ou_int)1 << 200, 0x1.0p+200L))
    return 1;

  // Large 256-bit value near max
  if (test__floatunoixf(
          make_oi(make_ti(0x023479FD0E092DC0LL, 0), make_ti(0, 0)),
          0x1.1A3CFE870496Ep+249L))
    return 1;

  // Max 256-bit unsigned value
  if (test__floatunoixf(
          make_oi(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                  make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)),
          0x1.0000000000000000p+256L))
    return 1;

#else
  printf("skipped\n");
#endif
  return 0;
}
