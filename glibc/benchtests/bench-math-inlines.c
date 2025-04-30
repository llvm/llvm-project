/* Measure math inline functions.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#define SIZE 1024
#define TEST_MAIN
#define TEST_NAME "math-inlines"
#define TEST_FUNCTION test_main ()
#include "bench-timing.h"
#include "json-lib.h"
#include "bench-util.h"

#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define BOOLTEST(func)					  \
static int __attribute__((noinline))			  \
func ## _f (double d, int i)				  \
{							  \
  if (func (d))						  \
    return (int) d + i;					  \
  else							  \
    return 5;						  \
}							  \
static int						  \
func ## _t (volatile double *p, size_t n, size_t iters)   \
{							  \
  int i, j;						  \
  int res = 0;						  \
  for (j = 0; j < iters; j++)				  \
    for (i = 0; i < n; i++)				  \
      if (func ## _f (p[i] * 2.0, i) != 0)		  \
	res += 5;					  \
  return res;						  \
}

#define VALUETEST(func)					  \
static int __attribute__((noinline))			  \
func ## _f (double d)					  \
{							  \
  return func (d);					  \
}							  \
static int						  \
func ## _t (volatile double *p, size_t n, size_t iters)	  \
{							  \
  int i, j;						  \
  int res = 0;						  \
  for (j = 0; j < iters; j++)				  \
    for (i = 0; i < n; i++)				  \
      res += func ## _f (p[i] * 2.0);			  \
  return res;						  \
}

typedef union
{
  double value;
  uint64_t word;
} ieee_double_shape_type;

#define EXTRACT_WORDS64(i,d)				  \
do {							  \
  ieee_double_shape_type gh_u;				  \
  gh_u.value = (d);					  \
  (i) = gh_u.word;					  \
} while (0)

/* Inlines similar to existing math_private.h versions.  */

static __always_inline int
__isnan_inl (double d)
{
  uint64_t di;
  EXTRACT_WORDS64 (di, d);
  return (di & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}

static __always_inline int
__isinf_ns2 (double d)
{
  uint64_t di;
  EXTRACT_WORDS64 (di, d);
  return (di & 0x7fffffffffffffffull) == 0x7ff0000000000000ull;
}

static __always_inline int
__finite_inl (double d)
{
  uint64_t di;
  EXTRACT_WORDS64 (di, d);
  return (di & 0x7fffffffffffffffull) < 0x7ff0000000000000ull;
}

#define __isnormal_inl(X) (__fpclassify (X) == FP_NORMAL)

/* Inlines for the builtin functions.  */

#define __isnan_builtin(X) __builtin_isnan (X)
#define __isinf_ns_builtin(X) __builtin_isinf (X)
#define __isinf_builtin(X) __builtin_isinf_sign (X)
#define __isfinite_builtin(X) __builtin_isfinite (X)
#define __isnormal_builtin(X) __builtin_isnormal (X)
#define __fpclassify_builtin(X) __builtin_fpclassify (FP_NAN, FP_INFINITE,  \
				  FP_NORMAL, FP_SUBNORMAL, FP_ZERO, (X))

static double __attribute ((noinline))
kernel_standard (double x, double y, int z)
{
  return x * y + z;
}

volatile double rem1 = 2.5;

static __always_inline double
remainder_test1 (double x)
{
  double y = rem1;
  if (((__builtin_expect (y == 0.0, 0) && !__isnan_inl (x))
	|| (__builtin_expect (__isinf_ns2 (x), 0) && !__isnan_inl (y))))
    return kernel_standard (x, y, 10);

  return remainder (x, y);
}

static __always_inline double
remainder_test2 (double x)
{
  double y = rem1;
  if (((__builtin_expect (y == 0.0, 0) && !__builtin_isnan (x))
	|| (__builtin_expect (__builtin_isinf (x), 0) && !__builtin_isnan (y))))
    return kernel_standard (x, y, 10);

  return remainder (x, y);
}

/* Create test functions for each possibility.  */

BOOLTEST (__isnan)
BOOLTEST (__isnan_inl)
BOOLTEST (__isnan_builtin)
BOOLTEST (isnan)

BOOLTEST (__isinf)
BOOLTEST (__isinf_builtin)
BOOLTEST (__isinf_ns2)
BOOLTEST (__isinf_ns_builtin)
BOOLTEST (isinf)

BOOLTEST (__finite)
BOOLTEST (__finite_inl)
BOOLTEST (__isfinite_builtin)
BOOLTEST (isfinite)

BOOLTEST (__isnormal_inl)
BOOLTEST (__isnormal_builtin)
BOOLTEST (isnormal)

VALUETEST (__fpclassify)
VALUETEST (__fpclassify_builtin)
VALUETEST (fpclassify)

VALUETEST (remainder_test1)
VALUETEST (remainder_test2)

typedef int (*proto_t) (volatile double *p, size_t n, size_t iters);

typedef struct
{
  const char *name;
  proto_t fn;
} impl_t;

#define IMPL(name) { #name, name ## _t }

static impl_t test_list[] =
{
  IMPL (__isnan),
  IMPL (__isnan_inl),
  IMPL (__isnan_builtin),
  IMPL (isnan),

  IMPL (__isinf),
  IMPL (__isinf_ns2),
  IMPL (__isinf_ns_builtin),
  IMPL (__isinf_builtin),
  IMPL (isinf),

  IMPL (__finite),
  IMPL (__finite_inl),
  IMPL (__isfinite_builtin),
  IMPL (isfinite),

  IMPL (__isnormal_inl),
  IMPL (__isnormal_builtin),
  IMPL (isnormal),

  IMPL (__fpclassify),
  IMPL (__fpclassify_builtin),
  IMPL (fpclassify),

  IMPL (remainder_test1),
  IMPL (remainder_test2)
};

static void
do_one_test (json_ctx_t *json_ctx, proto_t test_fn, volatile double *arr,
	     size_t len, const char *testname)
{
  size_t iters = 2048;
  timing_t start, stop, cur;

  json_attr_object_begin (json_ctx, testname);

  TIMING_NOW (start);
  test_fn (arr, len, iters);
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  json_attr_double (json_ctx, "duration", cur);
  json_attr_double (json_ctx, "iterations", iters);
  json_attr_double (json_ctx, "mean", cur / iters);
  json_attr_object_end (json_ctx);
}

static volatile double arr1[SIZE];
static volatile double arr2[SIZE];

int
test_main (void)
{
  json_ctx_t json_ctx;
  size_t i;

  bench_start ();

  json_init (&json_ctx, 2, stdout);
  json_attr_object_begin (&json_ctx, TEST_NAME);

  /* Create 2 test arrays, one with 10% zeroes, 10% negative values,
     79% positive values and 1% infinity/NaN.  The other contains
     50% inf, 50% NaN.  This relies on rand behaving correctly.  */

  for (i = 0; i < SIZE; i++)
    {
      int x = rand () & 255;
      arr1[i] = (x < 25) ? 0.0 : ((x < 50) ? -1 : 100);
      if (x == 255) arr1[i] = __builtin_inf ();
      if (x == 254) arr1[i] = __builtin_nan ("0");
      arr2[i] = (x < 128) ? __builtin_inf () : __builtin_nan ("0");
    }

  for (i = 0; i < sizeof (test_list) / sizeof (test_list[0]); i++)
    {
      json_attr_object_begin (&json_ctx, test_list[i].name);
      do_one_test (&json_ctx, test_list[i].fn, arr2, SIZE, "inf/nan");
      json_attr_object_end (&json_ctx);
    }

  for (i = 0; i < sizeof (test_list) / sizeof (test_list[0]); i++)
    {
      json_attr_object_begin (&json_ctx, test_list[i].name);
      do_one_test (&json_ctx, test_list[i].fn, arr1, SIZE, "normal");
      json_attr_object_end (&json_ctx);
    }

  json_attr_object_end (&json_ctx);
  return 0;
}

#include "bench-util.c"
#include "../test-skeleton.c"
