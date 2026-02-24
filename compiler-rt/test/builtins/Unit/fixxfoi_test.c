// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixxfoi
// REQUIRES: x86-target-arch
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#if defined(CRT_HAS_256BIT) && HAS_80_BIT_LONG_DOUBLE

// Returns: convert a to a signed 256-bit integer, rounding toward zero.

// Assumption: long double is an intel 80 bit floating point type padded with 6
// bytes
//             oi_int is a 256 bit integral type
//             value in long double is representable in oi_int (no range
//             checking performed)

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee
// eeee | 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm
// mmmm mmmm mmmm

COMPILER_RT_ABI oi_int __fixxfoi(long double a);

int test__fixxfoi(long double a, oi_int expected) {
  oi_int x = __fixxfoi(a);
  if (x != expected) {
    printf("error in __fixxfoi(%LA)\n", a);
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};
char assumption_2[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main() {
#if defined(CRT_HAS_256BIT) && HAS_80_BIT_LONG_DOUBLE
  if (test__fixxfoi(0.0, 0))
    return 1;

  if (test__fixxfoi(0.5, 0))
    return 1;
  if (test__fixxfoi(0.99, 0))
    return 1;
  if (test__fixxfoi(1.0, 1))
    return 1;
  if (test__fixxfoi(1.5, 1))
    return 1;
  if (test__fixxfoi(1.99, 1))
    return 1;
  if (test__fixxfoi(2.0, 2))
    return 1;
  if (test__fixxfoi(2.01, 2))
    return 1;
  if (test__fixxfoi(-0.5, 0))
    return 1;
  if (test__fixxfoi(-0.99, 0))
    return 1;
  if (test__fixxfoi(-1.0, -1))
    return 1;
  if (test__fixxfoi(-1.5, -1))
    return 1;
  if (test__fixxfoi(-1.99, -1))
    return 1;
  if (test__fixxfoi(-2.0, -2))
    return 1;
  if (test__fixxfoi(-2.01, -2))
    return 1;

  // Float precision boundary tests (from 128-bit reference)
  if (test__fixxfoi(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
    return 1;
  if (test__fixxfoi(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
    return 1;

  if (test__fixxfoi(-0x1.FFFFFEp+62, -(oi_int)0x7FFFFF8000000000LL))
    return 1;
  if (test__fixxfoi(-0x1.FFFFFCp+62, -(oi_int)0x7FFFFF0000000000LL))
    return 1;

  // Double precision boundary tests
  if (test__fixxfoi(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
    return 1;
  if (test__fixxfoi(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
    return 1;

  if (test__fixxfoi(-0x1.FFFFFFFFFFFFFp+62, -(oi_int)0x7FFFFFFFFFFFFC00LL))
    return 1;
  if (test__fixxfoi(-0x1.FFFFFFFFFFFFEp+62, -(oi_int)0x7FFFFFFFFFFFF800LL))
    return 1;

  // Long double (80-bit) full precision tests
  if (test__fixxfoi(0x1.FFFFFFFFFFFFFFFCp+62L, 0x7FFFFFFFFFFFFFFFLL))
    return 1;
  if (test__fixxfoi(0x1.FFFFFFFFFFFFFFF8p+62L, 0x7FFFFFFFFFFFFFFELL))
    return 1;

  if (test__fixxfoi(-0x1.0000000000000000p+63L, -(oi_int)0x8000000000000000LL))
    return 1;
  if (test__fixxfoi(-0x1.FFFFFFFFFFFFFFFCp+62L, -(oi_int)0x7FFFFFFFFFFFFFFFLL))
    return 1;
  if (test__fixxfoi(-0x1.FFFFFFFFFFFFFFF8p+62L, -(oi_int)0x7FFFFFFFFFFFFFFELL))
    return 1;

  // Tests at 128-bit boundary (same as ti tests, but still fits in oi)
  if (test__fixxfoi(0x1.FFFFFFFFFFFFFFFEp+126L,
                    make_oi(make_ti(0, 0), make_ti(0x7FFFFFFFFFFFFFFFLL,
                                                   0x8000000000000000LL))))
    return 1;
  if (test__fixxfoi(0x1.FFFFFFFFFFFFFFFCp+126L,
                    make_oi(make_ti(0, 0), make_ti(0x7FFFFFFFFFFFFFFFLL, 0))))
    return 1;

  if (test__fixxfoi(-0x1.0000000000000000p+127L,
                    -make_oi(make_ti(0, 0), make_ti(0x8000000000000000LL, 0))))
    return 1;

  // Tests beyond 128-bit boundary: values needing >128 bits
  // 2^200
  if (test__fixxfoi(0x1.0p+200L, (oi_int)1 << 200))
    return 1;
  if (test__fixxfoi(-0x1.0p+200L, -((oi_int)1 << 200)))
    return 1;

  // Value near 256-bit boundary
  // 0x1.FFFFFFFFFFFFFFFEp+254L is the largest xf value that fits in oi_int
  if (test__fixxfoi(0x1.FFFFFFFFFFFFFFFEp+254L,
                    make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, 0x8000000000000000LL),
                            make_ti(0, 0))))
    return 1;
  if (test__fixxfoi(
          -0x1.FFFFFFFFFFFFFFFEp+254L,
          -make_oi(make_ti(0x7FFFFFFFFFFFFFFFLL, 0x8000000000000000LL),
                   make_ti(0, 0))))
    return 1;

#else
  printf("skipped\n");
#endif
  return 0;
}
