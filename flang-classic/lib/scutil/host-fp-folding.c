/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Implement floating-point folding with host arithmetic
 *
 *  Implements the compile-time evaluation interfaces of "fp-folding.h"
 *  using the native host floating-point arithmetic operations and
 *  math library.
 */

#include "fp-folding.h"
#include <assert.h>
#include <complex.h>
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN64) && !defined(_M_ARM64)
#include <mmintrin.h>
#endif

/*
 *  Build-time sanity checks
 */
#if __STDC_VERSION__+0 < 199901
# warning C99 compiler required but __STDC_VERSION__ is less than 199901
#endif
#if FLT_RADIX+0 != 2
# error FLT_RADIX != 2
#endif
#if FLT_MANT_DIG+0 != 24
# error FLT_MANT_DIG != 24
#endif
#if DBL_MANT_DIG+0 != 53
# error DBL_MANT_DIG != 53
#endif
#if DECIMAL_DIG < 17
# error DECIMAL_DIG < 17
#endif

void
fold_sanity_check(void)
{
  assert(sizeof(float32_t) == 4);
  assert(sizeof(float64_t) == 8);
  assert(sizeof(float128_t) == 16);
}

/*
 *  C99 feature: alert the compiler that the following code cares about
 *  the dynamic floating-point environment.
 */
#ifndef __aarch64__
#pragma STDC FENV_ACCESS ON
#endif

/*
 *  Configure "denormal inputs are zero" and "flush denormal results to zero"
 *  modes.
 *
 *  TODO: Find a more portable way to configure these settings.
 */

static void
configure_denormals(bool denorms_are_zeros, bool flush_to_zero)
{
  fenv_t fenv;
  if (fegetenv(&fenv) != 0)
    fprintf(stderr, "fegetenv() failed: %s\n", strerror(errno));
#ifdef __x86_64__
#ifdef _WIN64
  unsigned int mxcsr = _mm_getcsr();
#else
  unsigned int mxcsr = fenv.__mxcsr;
#endif
  mxcsr &= ~0x0040;
  if (denorms_are_zeros)
    mxcsr |= 0x0040;
  mxcsr &= ~0x8000;
  if (flush_to_zero)
    mxcsr |= 0x8000;
#ifdef _WIN64
  _mm_setcsr( mxcsr );
#else
  fenv.__mxcsr = mxcsr;
#endif
#endif
#ifndef _WIN64
  if (fesetenv(&fenv) != 0)
    fprintf(stderr, "fesetenv() failed: %s\n", strerror(errno));
#endif
}

/*
 *  Comparisons.  These can't trap.
 */

#define COMPARE_BODY { \
  enum fold_relation rel = FOLD_UN; \
  fenv_t fenv; \
  if (feholdexcept(&fenv) != 0) \
    fprintf(stderr, "feholdexcept() failed: %s\n", strerror(errno)); \
  configure_denormals(false, false); \
  if (*x < *y) \
    rel = FOLD_LT; \
  else if (*x == *y) \
    rel = FOLD_EQ; \
  else if (*x > *y) \
    rel = FOLD_GT; \
  if (fesetenv(&fenv) != 0) \
    fprintf(stderr, "fesetenv() failed: %s\n", strerror(errno)); \
  return rel; \
}

enum fold_relation
fold_real32_compare(const float32_t *x, const float32_t *y)
{
  COMPARE_BODY
}

enum fold_relation
fold_real64_compare(const float64_t *x, const float64_t *y)
{
  COMPARE_BODY
}

enum fold_relation
fold_real128_compare(const float128_t *x, const float128_t *y)
{
  COMPARE_BODY
}

/*
 *  Set up the floating-point environment so that exceptional conditions
 *  are recorded but don't raise traps.
 *
 *  TODO: Consider setting the daz and flushz modes by the compiler to
 *  match the expected run-time environment (-Mdaz, -Mflushz).  Those
 *  compiler options are documented as "link-time" but maybe it would
 *  be a good idea to respect them while folding.
 */
static void
set_up_floating_point_environment(fenv_t *fenv)
{
  if (feholdexcept(fenv) != 0)
    fprintf(stderr, "feholdexcept() failed: %s\n", strerror(errno));
  configure_denormals(false, false);
  errno = 0;
}

/*
 *  Map floating-point exceptional conditions to a folding status code.
 */

static enum fold_status
interpret_exceptions(int exceptions, int errno_capture)
{
  if (exceptions & FE_INVALID)
    return FOLD_INVALID;
  if (exceptions & (FE_DIVBYZERO | FE_OVERFLOW))
    return FOLD_OVERFLOW;
  if (exceptions & FE_UNDERFLOW)
    return FOLD_UNDERFLOW;
  if (exceptions & FE_INEXACT)
    return FOLD_INEXACT;
  /* ignore any non-standard extended flags */
  if (errno_capture == EDOM)
    return FOLD_INVALID;
  if (errno_capture == ERANGE)
    return FOLD_OVERFLOW; /* can't distinguish over/underflow from errno */
  return FOLD_OK;
}

/*
 *  Common exit processing: restore the processor's floating-point
 *  environment and translate any exceptions that may have arisen.
 */

static enum fold_status
check_and_restore_floating_point_environment(const fenv_t *saved_fenv)
{
  int errno_capture = errno;
  int exceptions = fetestexcept(FE_ALL_EXCEPT);
  if (fesetenv(saved_fenv) != 0)
    fprintf(stderr, "fesetenv() failed: %s\n", strerror(errno));
  return interpret_exceptions(exceptions, errno_capture);
}

/*
 *  Compile-time evaluation routines for a large set of operations
 *  on all of the available host floating-point types.
 */

/* 32-bit */

enum fold_status
fold_int32_from_real32(int32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_int64_from_real32(int64_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_uint64_from_real32(uint64_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_uint32_from_real32(uint32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_from_int64(float32_t *res, const int64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_from_uint64(float32_t *res, const uint64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_from_real64(float32_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_from_real128(float32_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_negate(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = -*arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_abs(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = fabsf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_sqrt(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = sqrtf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_add(float32_t *res, const float32_t *x, const float32_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x+*y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_subtract(float32_t *res, const float32_t *x, const float32_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x - *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_multiply(float32_t *res, const float32_t *x, const float32_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x * *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_divide(float32_t *res, const float32_t *x, const float32_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x / *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_pow(float32_t *res, const float32_t *x, const float32_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = powf(*x, *y);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_sin(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = sinf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_cos(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = cosf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_tan(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = tanf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_asin(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = asinf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_acos(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = acosf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_atan(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = atanf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_atan2(float32_t *res, const float32_t *x, const float32_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = atan2f(*x, *y);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_exp(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = expf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_log(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = logf(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real32_log10(float32_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = log10f(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

/* 64-bit */

enum fold_status
fold_int32_from_real64(int32_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_int64_from_real64(int64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_uint32_from_real64(uint32_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_uint64_from_real64(uint64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_from_int64(float64_t *res, const int64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_from_uint64(float64_t *res, const uint64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_from_real32(float64_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_from_real128(float64_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_negate(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = -*arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_abs(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = fabs(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_sqrt(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = sqrt(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_add(float64_t *res, const float64_t *x, const float64_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x + *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_subtract(float64_t *res, const float64_t *x, const float64_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x - *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_multiply(float64_t *res, const float64_t *x, const float64_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x * *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_divide(float64_t *res, const float64_t *x, const float64_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x / *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_pow(float64_t *res, const float64_t *x, const float64_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = pow(*x, *y);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_sin(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = sin(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_cos(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = cos(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_tan(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = tan(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_asin(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = asin(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_acos(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = acos(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_atan(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = atan(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_atan2(float64_t *res, const float64_t *x, const float64_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = atan2(*x, *y);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_exp(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = exp(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_log(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = log(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real64_log10(float64_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = log10(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

/* 80, 64+64, or 128-bit */

enum fold_status
fold_int32_from_real128(int32_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_int64_from_real128(int64_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_uint32_from_real128(uint32_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_uint64_from_real128(uint64_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_from_int64(float128_t *res, const int64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_from_uint64(float128_t *res, const uint64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_from_real32(float128_t *res, const float32_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_from_real64(float128_t *res, const float64_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_negate(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = -*arg;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_abs(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = fabsl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_sqrt(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = sqrtl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_add(float128_t *res, const float128_t *x, const float128_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x + *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_subtract(float128_t *res, const float128_t *x, const float128_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x - *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_multiply(float128_t *res, const float128_t *x, const float128_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x * *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_divide(float128_t *res, const float128_t *x, const float128_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = *x / *y;
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_pow(float128_t *res, const float128_t *x, const float128_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = powl(*x, *y);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_sin(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = sinl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_cos(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = cosl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_tan(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = tanl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_asin(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = asinl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_acos(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = acosl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_atan(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = atanl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_atan2(float128_t *res, const float128_t *x, const float128_t *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = atan2l(*x, *y);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_exp(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = expl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_log(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = logl(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_real128_log10(float128_t *res, const float128_t *arg)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = log10l(*arg);
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_complex32_pow(FLOAT_COMPLEX_TYPE *res, const FLOAT_COMPLEX_TYPE *x, const FLOAT_COMPLEX_TYPE *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  /* use 'cpow' to improve precision of result. */
  *res = DCMPLX_TO_FCMPLX(cpow(FCMPLX_TO_DCMPLX(*x), FCMPLX_TO_DCMPLX(*y)));
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_complex64_pow(DOUBLE_COMPLEX_TYPE *res, const DOUBLE_COMPLEX_TYPE *x, const DOUBLE_COMPLEX_TYPE *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  /* use 'cpowl' to improve precision of result. */
  *res = LCMPLX_TO_DCMPLX(cpowl(DCMPLX_TO_LCMPLX(*x), DCMPLX_TO_LCMPLX(*y)));
  return check_and_restore_floating_point_environment(&saved_fenv);
}

enum fold_status
fold_complex128_pow(LONG_DOUBLE_COMPLEX_TYPE *res, const LONG_DOUBLE_COMPLEX_TYPE *x, const LONG_DOUBLE_COMPLEX_TYPE *y)
{
  fenv_t saved_fenv;
  set_up_floating_point_environment(&saved_fenv);
  *res = cpowl(*x, *y);
  return check_and_restore_floating_point_environment(&saved_fenv);
}
