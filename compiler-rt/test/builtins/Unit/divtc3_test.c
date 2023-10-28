// RUN: %clang_builtins %s %librt -lm -o %t && %run %t
// REQUIRES: librt_has_divtc3
// REQUIRES: c99-complex

//
// Bug 42493
// XFAIL: sparc-target-arch
//
#include <stdio.h>

#include "int_lib.h"
#include "int_math.h"
#include <complex.h>
#include <math.h>

// Returns: the quotient of (a + ib) / (c + id)

COMPILER_RT_ABI Qcomplex __divtc3(tf_float __a, tf_float __b, tf_float __c,
                                  tf_float __d);

enum {zero, non_zero, inf, NaN, non_zero_nan};

static int classify(Qcomplex x) {
  tf_float real = COMPLEXTF_REAL(x);
  tf_float imag = COMPLEXTF_IMAGINARY(x);
  if (real == 0.0 && imag == 0.0)
    return zero;
  if (crt_isinf(real) || crt_isinf(imag))
    return inf;
  if (crt_isnan(real) && crt_isnan(imag))
    return NaN;
  if (crt_isnan(real)) {
    if (imag == 0.0)
      return NaN;
    return non_zero_nan;
  }
  if (crt_isnan(imag)) {
    if (real == 0.0)
      return NaN;
    return non_zero_nan;
  }
  return non_zero;
}

static int test__divtc3(tf_float a, tf_float b, tf_float c, tf_float d) {
  Qcomplex r = __divtc3(a, b, c, d);
  Qcomplex dividend;
  Qcomplex divisor;

  COMPLEXTF_REAL(dividend) = a;
  COMPLEXTF_IMAGINARY(dividend) = b;
  COMPLEXTF_REAL(divisor) = c;
  COMPLEXTF_IMAGINARY(divisor) = d;

  switch (classify(dividend)) {
  case zero:
    switch (classify(divisor)) {
    case zero:
      if (classify(r) != NaN)
        return 1;
      break;
    case non_zero:
      if (classify(r) != zero)
        return 1;
      break;
    case inf:
      if (classify(r) != zero)
        return 1;
      break;
    case NaN:
      if (classify(r) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (classify(r) != NaN)
        return 1;
      break;
    }
    break;
  case non_zero:
    switch (classify(divisor)) {
    case zero:
      if (classify(r) != inf)
        return 1;
      break;
    case non_zero:
      if (classify(r) != non_zero)
        return 1;
      {
        tf_float zReal = (a * c + b * d) / (c * c + d * d);
        tf_float zImag = (b * c - a * d) / (c * c + d * d);
        Qcomplex diff =
            __divtc3(COMPLEXTF_REAL(r) - zReal, COMPLEXTF_IMAGINARY(r) - zImag,
                     COMPLEXTF_REAL(r), COMPLEXTF_IMAGINARY(r));
        // cabsl(z) == hypotl(creall(z), cimagl(z))
#ifdef CRT_LDBL_128BIT
        if (hypotl(COMPLEXTF_REAL(diff), COMPLEXTF_IMAGINARY(diff)) > 1.e-6)
#else
        // Avoid dependency on __trunctfxf2 for ld80 platforms and use double instead.
        if (hypot(COMPLEXTF_REAL(diff), COMPLEXTF_IMAGINARY(diff)) > 1.e-6)
#endif
          return 1;
      }
      break;
    case inf:
      if (classify(r) != zero)
        return 1;
      break;
    case NaN:
      if (classify(r) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (classify(r) != NaN)
        return 1;
      break;
    }
    break;
  case inf:
    switch (classify(divisor)) {
    case zero:
      if (classify(r) != inf)
        return 1;
      break;
    case non_zero:
      if (classify(r) != inf)
        return 1;
      break;
    case inf:
      if (classify(r) != NaN)
        return 1;
      break;
    case NaN:
      if (classify(r) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (classify(r) != NaN)
        return 1;
      break;
    }
    break;
  case NaN:
    switch (classify(divisor)) {
    case zero:
      if (classify(r) != NaN)
        return 1;
      break;
    case non_zero:
      if (classify(r) != NaN)
        return 1;
      break;
    case inf:
      if (classify(r) != NaN)
        return 1;
      break;
    case NaN:
      if (classify(r) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (classify(r) != NaN)
        return 1;
      break;
    }
    break;
  case non_zero_nan:
    switch (classify(divisor)) {
    case zero:
      if (classify(r) != inf)
        return 1;
      break;
    case non_zero:
      if (classify(r) != NaN)
        return 1;
      break;
    case inf:
      if (classify(r) != NaN)
        return 1;
      break;
    case NaN:
      if (classify(r) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (classify(r) != NaN)
        return 1;
      break;
    }
    break;
  }

  return 0;
}

tf_float x[][2] = {{1.e-6, 1.e-6},
                   {-1.e-6, 1.e-6},
                   {-1.e-6, -1.e-6},
                   {1.e-6, -1.e-6},

                   {1.e+6, 1.e-6},
                   {-1.e+6, 1.e-6},
                   {-1.e+6, -1.e-6},
                   {1.e+6, -1.e-6},

                   {1.e-6, 1.e+6},
                   {-1.e-6, 1.e+6},
                   {-1.e-6, -1.e+6},
                   {1.e-6, -1.e+6},

                   {1.e+6, 1.e+6},
                   {-1.e+6, 1.e+6},
                   {-1.e+6, -1.e+6},
                   {1.e+6, -1.e+6},

                   {NAN, NAN},
                   {-INFINITY, NAN},
                   {-2, NAN},
                   {-1, NAN},
                   {-0.5, NAN},
                   {-0., NAN},
                   {+0., NAN},
                   {0.5, NAN},
                   {1, NAN},
                   {2, NAN},
                   {INFINITY, NAN},

                   {NAN, -INFINITY},
                   {-INFINITY, -INFINITY},
                   {-2, -INFINITY},
                   {-1, -INFINITY},
                   {-0.5, -INFINITY},
                   {-0., -INFINITY},
                   {+0., -INFINITY},
                   {0.5, -INFINITY},
                   {1, -INFINITY},
                   {2, -INFINITY},
                   {INFINITY, -INFINITY},

                   {NAN, -2},
                   {-INFINITY, -2},
                   {-2, -2},
                   {-1, -2},
                   {-0.5, -2},
                   {-0., -2},
                   {+0., -2},
                   {0.5, -2},
                   {1, -2},
                   {2, -2},
                   {INFINITY, -2},

                   {NAN, -1},
                   {-INFINITY, -1},
                   {-2, -1},
                   {-1, -1},
                   {-0.5, -1},
                   {-0., -1},
                   {+0., -1},
                   {0.5, -1},
                   {1, -1},
                   {2, -1},
                   {INFINITY, -1},

                   {NAN, -0.5},
                   {-INFINITY, -0.5},
                   {-2, -0.5},
                   {-1, -0.5},
                   {-0.5, -0.5},
                   {-0., -0.5},
                   {+0., -0.5},
                   {0.5, -0.5},
                   {1, -0.5},
                   {2, -0.5},
                   {INFINITY, -0.5},

                   {NAN, -0.},
                   {-INFINITY, -0.},
                   {-2, -0.},
                   {-1, -0.},
                   {-0.5, -0.},
                   {-0., -0.},
                   {+0., -0.},
                   {0.5, -0.},
                   {1, -0.},
                   {2, -0.},
                   {INFINITY, -0.},

                   {NAN, 0.},
                   {-INFINITY, 0.},
                   {-2, 0.},
                   {-1, 0.},
                   {-0.5, 0.},
                   {-0., 0.},
                   {+0., 0.},
                   {0.5, 0.},
                   {1, 0.},
                   {2, 0.},
                   {INFINITY, 0.},

                   {NAN, 0.5},
                   {-INFINITY, 0.5},
                   {-2, 0.5},
                   {-1, 0.5},
                   {-0.5, 0.5},
                   {-0., 0.5},
                   {+0., 0.5},
                   {0.5, 0.5},
                   {1, 0.5},
                   {2, 0.5},
                   {INFINITY, 0.5},

                   {NAN, 1},
                   {-INFINITY, 1},
                   {-2, 1},
                   {-1, 1},
                   {-0.5, 1},
                   {-0., 1},
                   {+0., 1},
                   {0.5, 1},
                   {1, 1},
                   {2, 1},
                   {INFINITY, 1},

                   {NAN, 2},
                   {-INFINITY, 2},
                   {-2, 2},
                   {-1, 2},
                   {-0.5, 2},
                   {-0., 2},
                   {+0., 2},
                   {0.5, 2},
                   {1, 2},
                   {2, 2},
                   {INFINITY, 2},

                   {NAN, INFINITY},
                   {-INFINITY, INFINITY},
                   {-2, INFINITY},
                   {-1, INFINITY},
                   {-0.5, INFINITY},
                   {-0., INFINITY},
                   {+0., INFINITY},
                   {0.5, INFINITY},
                   {1, INFINITY},
                   {2, INFINITY},
                   {INFINITY, INFINITY}

};

int main() {
  const unsigned N = sizeof(x) / sizeof(x[0]);
  unsigned i, j;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      if (test__divtc3(x[i][0], x[i][1], x[j][0], x[j][1])) {
        fprintf(stderr, "Failed for %g, %g, %g, %g\n", (double)x[i][0],
                (double)x[i][1], (double)x[j][0], (double)x[j][1]);
        return 1;
      }
    }
  }

  fprintf(stderr, "No errors found.\n");
  return 0;
}
