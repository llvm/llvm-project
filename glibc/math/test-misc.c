/* Miscellaneous tests which don't fit anywhere else.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv.h>
#include <float.h>
#include <ieee754.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <math-tests.h>


static int
do_test (void)
{
  int result = 0;

#if LDBL_MANT_DIG > DBL_MANT_DIG
  {
    long double x = 0x100000001ll + (long double) 0.5;
    long double q;
    long double r;

    r = modfl (x, &q);
    if (q != (long double) 0x100000001ll || r != 0.5)
      {
	printf ("modfl (%Lg, ...) failed\n", x);
	result = 1;
      }
  }

  {
    long double x;
    long double m;
    long double r;
    int e;
    int i;

# if LDBL_MANT_DIG == 64
    m = 0xf.fffffffffffffffp-4L;
# elif LDBL_MANT_DIG == 106
    /* This has to match the mantissa of LDBL_MAX which actually does have a
       missing bit in the middle.  */
    m = 0x1.fffffffffffff7ffffffffffff8p-1L;
# elif LDBL_MANT_DIG == 113
    m = 0x1.ffffffffffffffffffffffffffffp-1L;
# else
#  error "Unsupported LDBL_MANT_DIG, please adjust"
# endif

    for (i = LDBL_MAX_EXP, x = LDBL_MAX; i >= LDBL_MIN_EXP; --i, x /= 2.0L)
      {
	printf ("2^%d: ", i);

	r = frexpl (x, &e);
	if (r != m)
	  {
	    printf ("mantissa incorrect: %.20La\n", r);
	    result = 1;
	    continue;
	  }
	if (e != i)
	  {
	    printf ("exponent wrong %d (%.20Lg)\n", e, x);
	    result = 1;
	    continue;
	  }
	puts ("ok");
      }

    for (i = LDBL_MIN_EXP, x = LDBL_MIN; i >= LDBL_MIN_EXP - LDBL_MANT_DIG + 1;
	 --i, x /= 2.0L)
      {
        printf ("2^%d: ", i);

        r = frexpl (x, &e);
        if (r != 0.5L)
          {
            printf ("mantissa incorrect: %.20La\n", r);
            result = 1;
            continue;
          }
        if (e != i)
          {
            printf ("exponent wrong %d (%.20Lg)\n", e, x);
            result = 1;
            continue;
          }
        puts ("ok");
      }

  }

# if 0
  {
    int e;
    long double r = frexpl (LDBL_MIN * LDBL_EPSILON, &e);

    if (r != 0.5)
      {
	printf ("frexpl (LDBL_MIN * LDBL_EPSILON, ...): mantissa wrong: %Lg\n",
		r);
	result = 1;
      }
    else if (e != -16444)
      {
	printf ("frexpl (LDBL_MIN * LDBL_EPSILON, ...): exponent wrong: %d\n",
		e);
	result = 1;
      }
  }
# endif
#endif

  {
    double x = 0x100000001ll + (double) 0.5;
    double q;
    double r;

    r = modf (x, &q);
    if (q != (double) 0x100000001ll || r != 0.5)
      {
	printf ("modf (%g, ...) failed\n", x);
	result = 1;
      }
  }

  {
    union ieee754_float v1;
    union ieee754_float v2;
    float f;

    v1.f = f = FLT_MIN;
    if (fpclassify (f) != FP_NORMAL)
      {
	printf ("fpclassify (FLT_MIN) failed: %d\n", fpclassify (f));
	result = 1;
      }
    f = nextafterf (f, FLT_MIN / 2.0f);
    if (fpclassify (f) != FP_SUBNORMAL)
      {
	printf ("fpclassify (FLT_MIN-epsilon) failed: %d\n", fpclassify (f));
	result = 1;
      }
    v2.f = f = nextafterf (f, FLT_MIN);
    if (fpclassify (f) != FP_NORMAL)
      {
	printf ("fpclassify (FLT_MIN-epsilon+epsilon) failed: %d\n",
		fpclassify (f));
	result = 1;
      }

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("FLT_MIN: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("FLT_MIN: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("FLT_MIN: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.f = f = -FLT_MIN;
    if (fpclassify (f) != FP_NORMAL)
      {
	printf ("fpclassify (-FLT_MIN) failed: %d\n", fpclassify (f));
	result = 1;
      }
    f = nextafterf (f, -FLT_MIN / 2.0f);
    if (fpclassify (f) != FP_SUBNORMAL)
      {
	printf ("fpclassify (-FLT_MIN-epsilon) failed: %d\n", fpclassify (f));
	result = 1;
      }
    v2.f = f = nextafterf (f, -FLT_MIN);
    if (fpclassify (f) != FP_NORMAL)
      {
	printf ("fpclassify (-FLT_MIN-epsilon+epsilon) failed: %d\n",
		fpclassify (f));
	result = 1;
      }

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("-FLT_MIN: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("-FLT_MIN: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("-FLT_MIN: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    f = FLT_MAX;
    if (fpclassify (f) != FP_NORMAL)
      {
	printf ("fpclassify (FLT_MAX) failed: %d\n", fpclassify (f));
	result = 1;
      }
    f = nextafterf (f, INFINITY);
    if (fpclassify (f) != FP_INFINITE)
      {
	printf ("fpclassify (FLT_MAX+epsilon) failed: %d\n", fpclassify (f));
	result = 1;
      }

    f = -FLT_MAX;
    if (fpclassify (f) != FP_NORMAL)
      {
	printf ("fpclassify (-FLT_MAX) failed: %d\n", fpclassify (f));
	result = 1;
      }
    f = nextafterf (f, -INFINITY);
    if (fpclassify (f) != FP_INFINITE)
      {
	printf ("fpclassify (-FLT_MAX-epsilon) failed: %d\n", fpclassify (f));
	result = 1;
      }

    v1.f = f = 0.0625;
    f = nextafterf (f, 0.0);
    v2.f = f = nextafterf (f, 1.0);

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("0.0625f down: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0625f down: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("0.0625f down: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.f = f = 0.0625;
    f = nextafterf (f, 1.0);
    v2.f = f = nextafterf (f, 0.0);

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("0.0625f up: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0625f up: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("0.0625f up: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.f = f = -0.0625;
    f = nextafterf (f, 0.0);
    v2.f = f = nextafterf (f, -1.0);

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("-0.0625f up: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("-0.0625f up: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("-0.0625f up: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.f = f = -0.0625;
    f = nextafterf (f, -1.0);
    v2.f = f = nextafterf (f, 0.0);

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("-0.0625f down: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("-0.0625f down: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("-0.0625f down: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.f = f = 0.0f;
    f = nextafterf (f, 1.0);
    v2.f = nextafterf (f, -1.0);

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("0.0f up: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0f up: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (0 != v2.ieee.negative)
      {
	printf ("0.0f up: negative differs: 0 vs %d\n",
		v2.ieee.negative);
	result = 1;
      }

    v1.f = f = 0.0f;
    f = nextafterf (f, -1.0);
    v2.f = nextafterf (f, 1.0);

    if (v1.ieee.mantissa != v2.ieee.mantissa)
      {
	printf ("0.0f down: mantissa differs: %8x vs %8x\n",
		v1.ieee.mantissa, v2.ieee.mantissa);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0f down: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (1 != v2.ieee.negative)
      {
	printf ("0.0f down: negative differs: 1 vs %d\n",
		v2.ieee.negative);
	result = 1;
      }

    if (nextafterf (0.0f, INFINITY) != nextafterf (0.0f, 1.0f)
        || nextafterf (-0.0f, INFINITY) != nextafterf (-0.0f, 1.0f)
        || nextafterf (0.0f, -INFINITY) != nextafterf (0.0f, -1.0f)
        || nextafterf (-0.0f, -INFINITY) != nextafterf (-0.0f, -1.0f))
      {
	printf ("nextafterf (+-0, +-Inf) != nextafterf (+-0, +-1)\n");
	result = 1;
      }

    if (nexttowardf (0.0f, INFINITY) != nexttowardf (0.0f, 1.0f)
        || nexttowardf (-0.0f, INFINITY) != nexttowardf (-0.0f, 1.0f)
        || nexttowardf (0.0f, -INFINITY) != nexttowardf (0.0f, -1.0f)
        || nexttowardf (-0.0f, -INFINITY) != nexttowardf (-0.0f, -1.0f))
      {
	printf ("nexttowardf (+-0, +-Inf) != nexttowardf (+-0, +-1)\n");
	result = 1;
      }
  }

  {
    union ieee754_double v1;
    union ieee754_double v2;
    double d;

    v1.d = d = DBL_MIN;
    if (fpclassify (d) != FP_NORMAL)
      {
	printf ("fpclassify (DBL_MIN) failed: %d\n", fpclassify (d));
	result = 1;
      }
    d = nextafter (d, DBL_MIN / 2.0);
    if (fpclassify (d) != FP_SUBNORMAL)
      {
	printf ("fpclassify (DBL_MIN-epsilon) failed: %d\n", fpclassify (d));
	result = 1;
      }
    v2.d = d = nextafter (d, DBL_MIN);
    if (fpclassify (d) != FP_NORMAL)
      {
	printf ("fpclassify (DBL_MIN-epsilon+epsilon) failed: %d\n",
		fpclassify (d));
	result = 1;
      }

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("DBL_MIN: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("DBL_MIN: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("DBL_MIN: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("DBL_MIN: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.d = d = -DBL_MIN;
    if (fpclassify (d) != FP_NORMAL)
      {
	printf ("fpclassify (-DBL_MIN) failed: %d\n", fpclassify (d));
	result = 1;
      }
    d = nextafter (d, -DBL_MIN / 2.0);
    if (fpclassify (d) != FP_SUBNORMAL)
      {
	printf ("fpclassify (-DBL_MIN-epsilon) failed: %d\n", fpclassify (d));
	result = 1;
      }
    v2.d = d = nextafter (d, -DBL_MIN);
    if (fpclassify (d) != FP_NORMAL)
      {
	printf ("fpclassify (-DBL_MIN-epsilon+epsilon) failed: %d\n",
		fpclassify (d));
	result = 1;
      }

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("-DBL_MIN: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("-DBL_MIN: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("-DBL_MIN: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("-DBL_MIN: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    d = DBL_MAX;
    if (fpclassify (d) != FP_NORMAL)
      {
	printf ("fpclassify (DBL_MAX) failed: %d\n", fpclassify (d));
	result = 1;
      }
    d = nextafter (d, INFINITY);
    if (fpclassify (d) != FP_INFINITE)
      {
	printf ("fpclassify (DBL_MAX+epsilon) failed: %d\n", fpclassify (d));
	result = 1;
      }

    d = -DBL_MAX;
    if (fpclassify (d) != FP_NORMAL)
      {
	printf ("fpclassify (-DBL_MAX) failed: %d\n", fpclassify (d));
	result = 1;
      }
    d = nextafter (d, -INFINITY);
    if (fpclassify (d) != FP_INFINITE)
      {
	printf ("fpclassify (-DBL_MAX-epsilon) failed: %d\n", fpclassify (d));
	result = 1;
      }

    v1.d = d = 0.0625;
    d = nextafter (d, 0.0);
    v2.d = d = nextafter (d, 1.0);

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("0.0625 down: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("0.0625 down: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0625 down: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("0.0625 down: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.d = d = 0.0625;
    d = nextafter (d, 1.0);
    v2.d = d = nextafter (d, 0.0);

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("0.0625 up: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("0.0625 up: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0625 up: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("0.0625 up: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.d = d = -0.0625;
    d = nextafter (d, 0.0);
    v2.d = d = nextafter (d, -1.0);

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("-0.0625 up: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("-0.0625 up: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("-0.0625 up: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("-0.0625 up: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.d = d = -0.0625;
    d = nextafter (d, -1.0);
    v2.d = d = nextafter (d, 0.0);

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("-0.0625 down: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("-0.0625 down: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("-0.0625 down: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (v1.ieee.negative != v2.ieee.negative)
      {
	printf ("-0.0625 down: negative differs: %d vs %d\n",
		v1.ieee.negative, v2.ieee.negative);
	result = 1;
      }

    v1.d = d = 0.0;
    d = nextafter (d, 1.0);
    v2.d = nextafter (d, -1.0);

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("0.0 up: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("0.0 up: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0 up: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (0 != v2.ieee.negative)
      {
	printf ("0.0 up: negative differs: 0 vs %d\n",
		v2.ieee.negative);
	result = 1;
      }

    v1.d = d = 0.0;
    d = nextafter (d, -1.0);
    v2.d = nextafter (d, 1.0);

    if (v1.ieee.mantissa0 != v2.ieee.mantissa0)
      {
	printf ("0.0 down: mantissa0 differs: %8x vs %8x\n",
		v1.ieee.mantissa0, v2.ieee.mantissa0);
	result = 1;
      }
    if (v1.ieee.mantissa1 != v2.ieee.mantissa1)
      {
	printf ("0.0 down: mantissa1 differs: %8x vs %8x\n",
		v1.ieee.mantissa1, v2.ieee.mantissa1);
	result = 1;
      }
    if (v1.ieee.exponent != v2.ieee.exponent)
      {
	printf ("0.0 down: exponent differs: %4x vs %4x\n",
		v1.ieee.exponent, v2.ieee.exponent);
	result = 1;
      }
    if (1 != v2.ieee.negative)
      {
	printf ("0.0 down: negative differs: 1 vs %d\n",
		v2.ieee.negative);
	result = 1;
      }

    if (nextafter (0.0, INFINITY) != nextafter (0.0, 1.0)
        || nextafter (-0.0, INFINITY) != nextafter (-0.0, 1.0)
        || nextafter (0.0, -INFINITY) != nextafter (0.0, -1.0)
        || nextafter (-0.0, -INFINITY) != nextafter (-0.0, -1.0))
      {
	printf ("nextafter (+-0, +-Inf) != nextafter (+-0, +-1)\n");
	result = 1;
      }

    if (nexttoward (0.0, INFINITY) != nexttoward (0.0, 1.0)
        || nexttoward (-0.0, INFINITY) != nexttoward (-0.0, 1.0)
        || nexttoward (0.0, -INFINITY) != nexttoward (0.0, -1.0)
        || nexttoward (-0.0, -INFINITY) != nexttoward (-0.0, -1.0))
      {
	printf ("nexttoward (+-0, +-Inf) != nexttoward (+-0, +-1)\n");
	result = 1;
      }
  }

#if LDBL_MANT_DIG > DBL_MANT_DIG
  {
    long double v1, v2;

    v1 = LDBL_MIN;
    if (fpclassify (v1) != FP_NORMAL)
      {
	printf ("fpclassify (LDBL_MIN) failed: %d (%La)\n",
		fpclassify (v1), v1);
	result = 1;
      }
    v2 = nextafterl (v1, LDBL_MIN / 2.0);
    if (fpclassify (v2) != FP_SUBNORMAL)
      {
	printf ("fpclassify (LDBL_MIN-epsilon) failed: %d (%La)\n",
		fpclassify (v2), v2);
	result = 1;
      }
    v2 = nextafterl (v2, LDBL_MIN);
    if (fpclassify (v2) != FP_NORMAL)
      {
	printf ("fpclassify (LDBL_MIN-epsilon+epsilon) failed: %d (%La)\n",
		fpclassify (v2), v2);
	result = 1;
      }

    if (v1 != v2)
      {
	printf ("LDBL_MIN-epsilon+epsilon != LDBL_MIN: %La vs %La\n", v2, v1);
	result = 1;
      }

    v1 = -LDBL_MIN;
    if (fpclassify (v1) != FP_NORMAL)
      {
	printf ("fpclassify (-LDBL_MIN) failed: %d (%La)\n",
		fpclassify (v1), v1);
	result = 1;
      }
    v2 = nextafterl (v1, -LDBL_MIN / 2.0);
    if (fpclassify (v2) != FP_SUBNORMAL)
      {
	printf ("fpclassify (-LDBL_MIN-epsilon) failed: %d (%La)\n",
		fpclassify (v2), v2);
	result = 1;
      }
    v2 = nextafterl (v2, -LDBL_MIN);
    if (fpclassify (v2) != FP_NORMAL)
      {
	printf ("fpclassify (-LDBL_MIN-epsilon+epsilon) failed: %d (%La)\n",
		fpclassify (v2), v2);
	result = 1;
      }

    if (v1 != v2)
      {
	printf ("-LDBL_MIN-epsilon+epsilon != -LDBL_MIN: %La vs %La\n", v2, v1);
	result = 1;
      }

    v1 = LDBL_MAX;
    if (fpclassify (v1) != FP_NORMAL)
      {
	printf ("fpclassify (LDBL_MAX) failed: %d (%La)\n",
		fpclassify (v1), v1);
	result = 1;
      }
    v2 = nextafterl (v1, INFINITY);
    if (fpclassify (v2) != FP_INFINITE)
      {
	printf ("fpclassify (LDBL_MAX+epsilon) failed: %d (%La)\n",
		fpclassify (v2), v2);
	result = 1;
      }

    v1 = -LDBL_MAX;
    if (fpclassify (v1) != FP_NORMAL)
      {
	printf ("fpclassify (-LDBL_MAX) failed: %d (%La)\n",
		fpclassify (v1), v1);
	result = 1;
      }
    v2 = nextafterl (v1, -INFINITY);
    if (fpclassify (v2) != FP_INFINITE)
      {
	printf ("fpclassify (-LDBL_MAX-epsilon) failed: %d (%La)\n",
		fpclassify (v2), v2);
	result = 1;
      }

    v1 = 0.0625;
    v2 = nextafterl (v1, 0.0);
    v2 = nextafterl (v2, 1.0);

    if (v1 != v2)
      {
	printf ("0.0625L-epsilon+epsilon != 0.0625L: %La vs %La\n", v2, v1);
	result = 1;
      }

    v1 = 0.0625;
    v2 = nextafterl (v1, 1.0);
    v2 = nextafterl (v2, 0.0);

    if (v1 != v2)
      {
	printf ("0.0625L+epsilon-epsilon != 0.0625L: %La vs %La\n", v2, v1);
	result = 1;
      }

    v1 = -0.0625;
    v2 = nextafterl (v1, 0.0);
    v2 = nextafterl (v2, -1.0);

    if (v1 != v2)
      {
	printf ("-0.0625L+epsilon-epsilon != -0.0625L: %La vs %La\n", v2, v1);
	result = 1;
      }

    v1 = -0.0625;
    v2 = nextafterl (v1, -1.0);
    v2 = nextafterl (v2, 0.0);

    if (v1 != v2)
      {
	printf ("-0.0625L-epsilon+epsilon != -0.0625L: %La vs %La\n", v2, v1);
	result = 1;
      }

    v1 = 0.0;
    v2 = nextafterl (v1, 1.0);
    v2 = nextafterl (v2, -1.0);

    if (v1 != v2)
      {
	printf ("0.0+epsilon-epsilon != 0.0L: %La vs %La\n", v2, v1);
	result = 1;
      }
    if (signbit (v2))
      {
	printf ("0.0+epsilon-epsilon is negative\n");
	result = 1;
      }

    v1 = 0.0;
    v2 = nextafterl (v1, -1.0);
    v2 = nextafterl (v2, 1.0);

    if (v1 != v2)
      {
	printf ("0.0-epsilon+epsilon != 0.0L: %La vs %La\n", v2, v1);
	result = 1;
      }
    if (!signbit (v2))
      {
	printf ("0.0-epsilon+epsilon is positive\n");
	result = 1;
      }

    if (nextafterl (0.0, INFINITY) != nextafterl (0.0, 1.0)
        || nextafterl (-0.0, INFINITY) != nextafterl (-0.0, 1.0)
        || nextafterl (0.0, -INFINITY) != nextafterl (0.0, -1.0)
        || nextafterl (-0.0, -INFINITY) != nextafterl (-0.0, -1.0))
      {
	printf ("nextafterl (+-0, +-Inf) != nextafterl (+-0, +-1)\n");
	result = 1;
      }

    if (nexttowardl (0.0L, INFINITY) != nexttowardl (0.0L, 1.0L)
        || nexttowardl (-0.0L, INFINITY) != nexttowardl (-0.0L, 1.0L)
        || nexttowardl (0.0L, -INFINITY) != nexttowardl (0.0L, -1.0L)
        || nexttowardl (-0.0L, -INFINITY) != nexttowardl (-0.0L, -1.0L))
      {
	printf ("nexttowardl (+-0, +-Inf) != nexttowardl (+-0, +-1)\n");
	result = 1;
      }
  }
#endif

  if (! isnormal (FLT_MIN))
    {
      puts ("isnormal (FLT_MIN) failed");
      result = 1;
    }
  if (! isnormal (DBL_MIN))
    {
      puts ("isnormal (DBL_MIN) failed");
      result = 1;
    }
#if LDBL_MANT_DIG > DBL_MANT_DIG
  if (! isnormal (LDBL_MIN))
    {
      puts ("isnormal (LDBL_MIN) failed");
      result = 1;
    }
#endif

#if defined (__i386__) || defined (__x86_64__)
  /* This is a test for the strange long doubles in x86 FPUs.  */
  {
    union
    {
      char b[10];
      long double d;
    } u =
      { .b = { 0, 0, 0, 0, 0, 0, 0, 0x80, 0, 0 } };

    if (fpclassify (u.d) != FP_NORMAL)
      {
	printf ("fpclassify (0x00008000000000000000) failed: %d (%Lg)\n",
		fpclassify (u.d), u.d);
	result = 1;
      }
  }

  /* Special qNaNs in x86 long double.  Test for scalbl.  */
  {
    union
    {
      char b[10];
      long double d;
    } u =
      { .b = { 0, 1, 0, 0, 0, 0, 0, 0xc0, 0xff, 0x7f } };
    long double r;

    r = scalbl (u.d, 0.0);
    if (!isnan (r))
      {
	puts ("scalbl (qNaN, 0) does not return NaN");
	result = 1;
      }
    else if (memcmp (&r, &u.d, sizeof (double)) != 0)
      {
	puts ("scalbl (qNaN, 0) does not return the same NaN");
	result = 1;
      }
  }
#endif

#if LDBL_MANT_DIG > DBL_MANT_DIG
  {
    long double r;

    feclearexcept (FE_ALL_EXCEPT);
    r = scalbl (LDBL_MIN, 2147483647);
    if (! isinf (r))
      {
	puts ("scalbl (LDBL_MIN, 2147483647) does not return Inf");
	result = 1;
      }
    else if (signbit (r) != 0)
      {
	puts ("scalbl (LDBL_MIN, 2147483647) returns -Inf");
	result = 1;
      }
# ifdef FE_UNDERFLOW
    else if (fetestexcept (FE_UNDERFLOW))
      {
	puts ("scalbl (LDBL_MIN, 2147483647) raises underflow exception");
	result = 1;
      }
# endif

    feclearexcept (FE_ALL_EXCEPT);
    r = scalbl (LDBL_MAX, -2147483647);
    if (r != 0.0)
      {
	puts ("scalbl (LDBL_MAX, -2147483647) does not return 0");
	result = 1;
      }
    else if (signbit (r) != 0)
      {
	puts ("scalbl (LDBL_MAX, -2147483647) returns -Inf");
	result = 1;
      }
# ifdef FE_OVERFLOW
    else if (fetestexcept (FE_OVERFLOW))
      {
	puts ("scalbl (LDBL_MAX, -2147483647) raises overflow exception");
	result = 1;
      }
# endif
  }
#endif

  /* The tests here are very similar to tests earlier in this file,
     the important difference is just that there are no intervening
     union variables that cause some GCC versions to hide possible
     bugs in nextafter* implementation.  */
  if (nextafterf (nextafterf (FLT_MIN, FLT_MIN / 2.0), FLT_MIN) != FLT_MIN)
    {
      puts ("nextafterf FLT_MIN test failed");
      result = 1;
    }
  if (nextafterf (nextafterf (-FLT_MIN, -FLT_MIN / 2.0), -FLT_MIN)
      != -FLT_MIN)
    {
      puts ("nextafterf -FLT_MIN test failed");
      result = 1;
    }
  if (nextafter (nextafter (DBL_MIN, DBL_MIN / 2.0), DBL_MIN) != DBL_MIN)
    {
      puts ("nextafter DBL_MIN test failed");
      result = 1;
    }
  if (nextafter (nextafter (-DBL_MIN, -DBL_MIN / 2.0), -DBL_MIN) != -DBL_MIN)
    {
      puts ("nextafter -DBL_MIN test failed");
      result = 1;
    }
#if LDBL_MANT_DIG > DBL_MANT_DIG
  if (nextafterl (nextafterl (LDBL_MIN, LDBL_MIN / 2.0), LDBL_MIN)
      != LDBL_MIN)
    {
      puts ("nextafterl LDBL_MIN test failed");
      result = 1;
    }
  if (nextafterl (nextafterl (-LDBL_MIN, -LDBL_MIN / 2.0), -LDBL_MIN)
      != -LDBL_MIN)
    {
      puts ("nextafterl -LDBL_MIN test failed");
      result = 1;
    }
#endif

  volatile float f1 = FLT_MAX;
  volatile float f2 = FLT_MAX / 2;
  (void) &f1;
  (void) &f2;
  feclearexcept (FE_ALL_EXCEPT);
  f2 += f1;
#if defined(FE_OVERFLOW) && defined(FE_INEXACT)
  int fe = fetestexcept (FE_ALL_EXCEPT);
  if (EXCEPTION_TESTS (float) && fe != (FE_OVERFLOW | FE_INEXACT))
    {
      printf ("float overflow test failed: %x\n", fe);
      result = 1;
    }
#endif

  volatile double d1 = DBL_MAX;
  volatile double d2 = DBL_MAX / 2;
  (void) &d1;
  (void) &d2;
  feclearexcept (FE_ALL_EXCEPT);
  d2 += d1;
#if defined(FE_OVERFLOW) && defined(FE_INEXACT)
  fe = fetestexcept (FE_ALL_EXCEPT);
  if (EXCEPTION_TESTS (double) && fe != (FE_OVERFLOW | FE_INEXACT))
    {
      printf ("double overflow test failed: %x\n", fe);
      result = 1;
    }
#endif

#if LDBL_MANT_DIG > DBL_MANT_DIG
  volatile long double ld1 = LDBL_MAX;
  volatile long double ld2 = LDBL_MAX / 2;
  (void) &ld1;
  (void) &ld2;
  feclearexcept (FE_ALL_EXCEPT);
  ld2 += ld1;
# if defined(FE_OVERFLOW) && defined(FE_INEXACT)
  fe = fetestexcept (FE_ALL_EXCEPT);
  if (EXCEPTION_TESTS (long double) && fe != (FE_OVERFLOW | FE_INEXACT))
    {
      printf ("long double overflow test failed: %x\n", fe);
      result = 1;
    }
# endif

# if LDBL_MANT_DIG == 113
  volatile long double ld3 = 0x1.0000000000010000000100000001p+1;
  volatile long double ld4 = 0x1.0000000000000000000000000001p+1;
  (void) &ld3;
  (void) &ld4;
  ld3 -= ld4;
  if (ld3 != 0x1.0p-47)
    {
      printf ("long double subtraction test failed %.28La\n", ld3);
      result = 1;
    }
# endif

/* Skip testing IBM long double format, for 2 reasons:
   1) it only supports FE_TONEAREST
   2) nextafter (0.0, 1.0) == nextafterl (0.0L, 1.0L), so
      nextafter (0.0, 1.0) / 16.0L will be 0.0L.  */
# if LDBL_MANT_DIG >= DBL_MANT_DIG + 4 && LDBL_MANT_DIG != 106
  int oldmode = fegetround ();
  int j;
  for (j = 0; j < 4; j++)
    {
      int mode;
      int i;
      int k = 0;
      const char *mstr;
      switch (j)
	{
#ifdef FE_TONEAREST
	case 0:
	  mode = FE_TONEAREST;
	  mstr = "nearest";
	  k = 8;
	  break;
#endif
#ifdef FE_DOWNWARD
	case 1:
	  mode = FE_DOWNWARD;
	  mstr = "-inf";
	  break;
#endif
#ifdef FE_UPWARD
	case 2:
	  mode = FE_UPWARD;
	  mstr = "+inf";
	  k = 15;
	  break;
#endif
#ifdef FE_TOWARDZERO
	case 3:
	  mode = FE_TOWARDZERO;
	  mstr = "0";
	  break;
#endif
	default:
	  continue;
	}

      volatile long double ld5 = nextafter (0.0, 1.0) / 16.0L;
      volatile double d5;
      (void) &ld5;
      for (i = 0; i <= 32; i++)
	{
	  if (fesetround (mode))
	    {
	      printf ("failed to set rounding mode to %s\n", mstr);
	      if (ROUNDING_TESTS (long double, mode)
		  && ROUNDING_TESTS (double, mode))
		result = 1;
	      else
		puts ("ignoring this failure");
	      break;
	    }
	  d5 = ld5 * i;
	  (void) &d5;
	  fesetround (oldmode);
	  if (d5 != ((j == 0 && i == 8) ? 0 : (i + k) / 16)
		    * nextafter (0.0, 1.0))
	    {
	      printf ("%La incorrectly rounded to %s as %a\n",
		      ld5 * i, mstr, d5);
	      if (ROUNDING_TESTS (long double, mode)
		  && ROUNDING_TESTS (double, mode))
		result = 1;
	      else
		puts ("ignoring this failure");
	    }
	}
    }

#  ifdef FE_UPWARD
  volatile long double ld7 = nextafterl (0.0L, 1.0L);
  volatile double d7;
  (void) &ld7;
  fesetround (FE_UPWARD);
  d7 = ld7;
  (void) &d7;
  fesetround (oldmode);

  if (d7 != nextafter (0.0, 1.0))
    {
      printf ("%La incorrectly rounded upward to %a\n", ld7, d7);
      if (ROUNDING_TESTS (long double, FE_UPWARD)
	  && ROUNDING_TESTS (double, FE_UPWARD))
	result = 1;
      else
	puts ("ignoring this failure");
    }
#  endif
# endif
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
