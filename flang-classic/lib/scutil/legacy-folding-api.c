/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Implement legacy folding interfaces
 *
 *  Implement the legacy compile-time evaluation library atop
 *  native host integer arithmetic, the floating-point
 *  folding package, and whatever 128-bit integer support
 *  has been supplied in int128.h.
 *
 *  These interfaces date back to the days when IEEE-754 floating-point
 *  arithmetic was a novelty and cross-compilation was a common practice.
 *  They were formerly implemented with an IEEE-754 arithmetic emulation
 *  software package that had been translated into C from its original
 *  PDP-11 assembly language.  These interfaces have been cleaned up
 *  to some degree, though more work remains to be done.  This particular
 *  implementation is new, and comprises mostly conversions between
 *  the operand and result types of these legacy interfaces and those
 *  of the underlying integer and floating-point folding packages.
 */

#include "legacy-folding-api.h"
#include "fp-folding.h"
#include "int128.h"
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static bool
is_host_little_endian(void)
{
  static const int one = 1;
  return *(const char *) &one != 0;
}

/*
 *  Conversions between legacy scutil API types and the standard C
 *  fixed-size integer types and the types of the floating-point
 *  folding package.
 *
 *  Pointer arguments are used here to enforce type safety and avoid
 *  silent conversions.
 */

static void
unwrap_l(int64_t *x, const DBLINT64 y)
{
  *x = (int64_t) y[0] << 32 | (int64_t) (uint32_t) y[1];
}

static void
unwrap_u(uint64_t *x, const DBLUINT64 y)
{
  *x = (uint64_t) y[0] << 32 | (uint64_t) (uint32_t) y[1];
}

static void
wrap_l(DBLINT64 res, int64_t *x)
{
  res[0] = *x >> 32; /* big end */
  res[1] = *x;
}

static void
wrap_u(DBLUINT64 res, uint64_t *x)
{
  res[0] = *x >> 32; /* big end */
  res[1] = *x;
}

static void
unwrap_f(float32_t *f, IEEE32 *x)
{
  *f = *(float32_t *) x;
}

static void
wrap_f(IEEE32 *f, float32_t *x)
{
  *f = *(IEEE32 *) x;
}

static void
unwrap_d(float64_t *x, const IEEE64 d)
{
  union {
    float64_t x;
    uint32_t i[2];
  } u;
  int le = is_host_little_endian();
  u.i[le] = d[0]; /* big end */
  u.i[le ^ 1] = d[1];
  *x = u.x;
}

static void
wrap_d(IEEE64 res, const float64_t *x)
{
  union {
    float64_t d;
    uint32_t i[2];
  } u;
  u.d = *x;
  int le = is_host_little_endian();
  res[0] = u.i[le]; /* big end */
  res[1] = u.i[le ^ 1];
}

/*
 *  Legacy conversion interfaces
 */

void
xdtomd(IEEE64 d, double *md)
{
  unwrap_d(md, d);
}

void
xmdtod(double md, IEEE64 d)
{
  wrap_d(d, &md);
}

/*
 *  64-bit integer operations
 */

int
cmp64(DBLINT64 arg1, DBLINT64 arg2)
{
  int64_t y, z;
  unwrap_l(&y, arg1);
  unwrap_l(&z, arg2);
  return y < z ? -1 : y > z;
}

int
ucmp64(DBLUINT64 arg1, DBLUINT64 arg2)
{
  uint64_t y, z;
  unwrap_u(&y, arg1);
  unwrap_u(&z, arg2);
  return y < z ? -1 : y > z;
}

void
add64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int64_t i1, i2, res;
  unwrap_l(&i1, arg1);
  unwrap_l(&i2, arg2);
  res = i1 + i2;
  wrap_l(result, &res);
}

void
div64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int64_t num, den, res;
  unwrap_l(&num, arg1);
  unwrap_l(&den, arg2);
  res = den == 0 ? 0 : num / den;
  wrap_l(result, &res);
}

void
exp64(DBLINT64 arg, INT exp, DBLINT64 result)
{
  int64_t base, x = exp >= 0;
  unwrap_l(&base, arg);
  while (exp-- > 0)
    x *= base;
  wrap_l(result, &x);
}

void
mul64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int64_t i1, i2, res;
  unwrap_l(&i1, arg1);
  unwrap_l(&i2, arg2);
  res = i1 * i2;
  wrap_l(result, &res);
}

void
mul64_10(DBLINT64 arg, DBLINT64 result)
{
  int64_t i;
  unwrap_l(&i, arg);
  i *= 10;
  wrap_l(result, &i);
}

void
neg64(DBLINT64 arg, DBLINT64 result)
{
  int64_t i;
  unwrap_l(&i, arg);
  i = -i;
  wrap_l(result, &i);
}

void
shf64(DBLINT64 arg, int count, DBLINT64 result)
{
  int64_t x;
  unwrap_l(&x, arg);
  if (count < -63 || count > 63)
    x = 0;
  else if (count <= 0)
    x >>= -count;
  else
    x <<= count;
  wrap_l(result, &x);
}

void
sub64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int64_t i1, i2, res;
  unwrap_l(&i1, arg1);
  unwrap_l(&i2, arg2);
  res = i1 - i2;
  wrap_l(result, &res);
}

void
uadd64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result)
{
  uint64_t u1, u2, res;
  unwrap_u(&u1, arg1);
  unwrap_u(&u2, arg2);
  res = u1 + u2;
  wrap_u(result, &res);
}

void
uneg64(DBLUINT64 arg, DBLUINT64 result)
{
  uint64_t u;
  unwrap_u(&u, arg);
  u = -u;
  wrap_u(result, &u);
}

void
ushf64(DBLUINT64 arg, int count, DBLUINT64 result)
{
  uint64_t u;
  unwrap_u(&u, arg);
  if (count < -63 || count > 63)
    u = 0;
  else if (count <= 0)
    u >>= -count;
  else
    u <<= count;
  wrap_u(result, &u);
}

void
usub64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result)
{
  uint64_t u1, u2, res;
  unwrap_u(&u1, arg1);
  unwrap_u(&u2, arg2);
  res = u1 - u2;
  wrap_u(result, &res);
}

void
udiv64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result)
{
  uint64_t num, den, res;
  unwrap_u(&num, arg1);
  unwrap_u(&den, arg2);
  res = den == 0 ? 0 : num / den;
  wrap_u(result, &res);
}

void
umul64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result)
{
  uint64_t u1, u2, res;
  unwrap_u(&u1, arg1);
  unwrap_u(&u2, arg2);
  res = u1 * u2;
  wrap_u(result, &res);
}

void
umul64_10(DBLUINT64 arg, DBLUINT64 result)
{
  uint64_t u;
  unwrap_u(&u, arg);
  u *= 10;
  wrap_u(result, &u);
}

void
and64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int64_t i1, i2, res;
  unwrap_l(&i1, arg1);
  unwrap_l(&i2, arg2);
  res = i1 & i2;
  wrap_l(result, &res);
}

void
not64(DBLINT64 arg, DBLINT64 result)
{
  int64_t i;
  unwrap_l(&i, arg);
  i = ~i;
  wrap_l(result, &i);
}

void
or64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int64_t i1, i2, res;
  unwrap_l(&i1, arg1);
  unwrap_l(&i2, arg2);
  res = i1 | i2;
  wrap_l(result, &res);
}

void
xor64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int64_t i1, i2, res;
  unwrap_l(&i1, arg1);
  unwrap_l(&i2, arg2);
  res = i1 ^ i2;
  wrap_l(result, &res);
}

void
ui64toax(DBLINT64 from, char *to, int size, int is_unsigned, int radix)
{
  int64_t x;
  unwrap_l(&x, from);
  if (!is_unsigned && x < 0) {
    *to++ = '-';
    --size;
    x = -x;
  }
  switch (radix) {
  case 8:
    snprintf(to, size, "%" PRIo64, x);
    break;
  case 10:
    snprintf(to, size, "%" PRIu64, x);
    break;
  case 16:
    snprintf(to, size, "%" PRIx64, x);
    break;
  default:
    assert(!"bad radix");
  }
}

/*
 *  128-bit integer routines
 *
 *  Note: if GCC >= 4.8 were guaranteed to be available, its
 *  built-in __int128 type could be used here explicitly.
 *  For flexibility, the wrapper API in int128.h is used.
 */

static void
unwrap_i128(int128_t *result, const INT128 i /* big-endian */)
{
  int j;
  int128_from_int64(result, 0);
  for (j = 0; j < 4; ++j) {
    int128_t part, shifted;
    int128_from_uint64(&part, (uint32_t) i[j ^ 3]);
    int128_shift_left(&shifted, &part, 32 * j);
    int128_or(result, result, &shifted);
  }
}

static void
wrap_i128(INT128 i, const int128_t *x)
{
  int j;
  for (j = 0; j < 4; ++j) {
    int128_t t;
    int64_t least;
    int128_shift_right_logical(&t, x, 32 * j);
    int128_to_int64(&least, &t);
    i[j ^ 3] = least;
  }
}

void
shf128(INT128 arg, int count, INT128 result)
{
  int128_t x, res;
  unwrap_i128(&x, arg);
  if (count < 0) {
    /* negative count means right shift */
    int128_shift_right_logical(&res, &x, -count);
  } else {
    int128_shift_left(&res, &x, count);
  }
  wrap_i128(result, &res);
}

void
add128(INT128 arg1, INT128 arg2, INT128 result)
{
  int128_t x, y, z;
  unwrap_i128(&x, arg1);
  unwrap_i128(&y, arg2);
  int128_unsigned_add(&z, &x, &y);
  wrap_i128(result, &z);
}

void
sub128(INT128 arg1, INT128 arg2, INT128 result)
{
  int128_t x, y, z;
  unwrap_i128(&x, arg1);
  unwrap_i128(&y, arg2);
  int128_signed_subtract(&z, &x, &y);
  wrap_i128(result, &z);
}

int
cmp128(INT128 arg1, INT128 arg2)
{
  int128_t x, y;
  unwrap_i128(&x, arg1);
  unwrap_i128(&y, arg2);
  return int128_signed_compare(&x, &y);
}

/* omitted: ucmp128, mul256, negN, shfN et al. */

void
mul128l(INT128 arg1, INT128 arg2, INT128 result)
{
  int128_t x, y, high, low;
  unwrap_i128(&x, arg1);
  unwrap_i128(&y, arg2);
  int128_signed_multiply(&high, &low, &x, &y);
  wrap_i128(result, &low);
}

void
div128(INT128 arg1, INT128 arg2, INT128 result)
{
  int128_t x, y, quotient, remainder;
  unwrap_i128(&x, arg1);
  unwrap_i128(&y, arg2);
  int128_signed_divide(&quotient, &remainder, &x, &y);
  wrap_i128(result, &quotient);
}

/*
 *  32-bit floating-point
 */

/* Check any error result from a fold_... routine and pass it on to
 * the user's fperror() if needed.
 */
static void
check(enum fold_status status)
{
  switch (status) {
  case FOLD_OK:
  case FOLD_INEXACT:
    break;
  case FOLD_INVALID:
    fperror(FPE_INVOP);
    break;
  case FOLD_OVERFLOW:
    fperror(FPE_FPOVF);
    break;
  case FOLD_UNDERFLOW:
    fperror(FPE_FPUNF);
    break;
  }
}

void
xfix(IEEE32 f, INT *i)
{
  float32_t x;
  unwrap_f(&x, &f);
  check(fold_int32_from_real32(i, &x));
}

void
xfixu(IEEE32 f, UINT *u)
{
  float32_t x;
  unwrap_f(&x, &f);
  check(fold_uint32_from_real32(u, &x));
}

void
xffixu(IEEE32 f, UINT *u)
{
  xfixu(f, u); /* are both xfixu and xffixu actually used? */
}

void
xfix64(IEEE32 f, DBLINT64 l)
{
  int64_t x;
  float32_t y;
  unwrap_f(&y, &f);
  check(fold_int64_from_real32(&x, &y));
  wrap_l(l, &x);
}

void
xfixu64(IEEE32 f, DBLUINT64 u)
{
  int64_t x;
  float32_t y;
  uint64_t xu;
  unwrap_f(&y, &f);
  check(fold_int64_from_real32(&x, &y));
  xu = x;
  wrap_u(u, &xu);
}

void
xffloat(INT i, IEEE32 *f)
{
  float32_t x;
  int64_t li = i;
  check(fold_real32_from_int64(&x, &li));
  wrap_f(f, &x);
}

void
xffloatu(UINT u, IEEE32 *f)
{
  float32_t x;
  int64_t li = u;
  check(fold_real32_from_int64(&x, &li));
  wrap_f(f, &x);
}

void
xflt64(DBLINT64 l, IEEE32 *f)
{
  float32_t x;
  int64_t y;
  unwrap_l(&y, l);
  check(fold_real32_from_int64(&x, &y));
  wrap_f(f, &x);
}

void
xfltu64(DBLUINT64 u, IEEE32 *f)
{
  float32_t x;
  uint64_t y;
  unwrap_u(&y, u);
  check(fold_real32_from_uint64(&x, &y));
  wrap_f(f, &x);
}

void
xfadd(IEEE32 f1, IEEE32 f2, IEEE32 *r)
{
  float32_t x, y, z;
  unwrap_f(&y, &f1);
  unwrap_f(&z, &f2);
  check(fold_real32_add(&x, &y, &z));
  wrap_f(r, &x);
}

void
xfsub(IEEE32 f1, IEEE32 f2, IEEE32 *r)
{
  float32_t x, y, z;
  unwrap_f(&y, &f1);
  unwrap_f(&z, &f2);
  check(fold_real32_subtract(&x, &y, &z));
  wrap_f(r, &x);
}

void
xfneg(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_negate(&x, &y));
  wrap_f(r, &x);
}

void
xfmul(IEEE32 f1, IEEE32 f2, IEEE32 *r)
{
  float32_t x, y, z;
  unwrap_f(&y, &f1);
  unwrap_f(&z, &f2);
  check(fold_real32_multiply(&x, &y, &z));
  wrap_f(r, &x);
}

void
xfdiv(IEEE32 f1, IEEE32 f2, IEEE32 *r)
{
  float32_t x, y, z;
  unwrap_f(&y, &f1);
  unwrap_f(&z, &f2);
  check(fold_real32_divide(&x, &y, &z));
  wrap_f(r, &x);
}

void
xfrcp(IEEE32 f, IEEE32 *r)
{
  float32_t x, one, y;
  int64_t ione = 1;
  check(fold_real32_from_int64(&one, &ione));
  unwrap_f(&y, &f);
  check(fold_real32_divide(&x, &one, &y));
  wrap_f(r, &x);
}

void
xfabsv(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_abs(&x, &y));
  wrap_f(r, &x);
}

void
xfsqrt(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_sqrt(&x, &y));
  wrap_f(r, &x);
}

void
xfpow(IEEE32 f1, IEEE32 f2, IEEE32 *r)
{
  float32_t x, y, z;
  unwrap_f(&y, &f1);
  unwrap_f(&z, &f2);
  check(fold_real32_pow(&x, &y, &z));
  wrap_f(r, &x);
}

void
xfsin(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_sin(&x, &y));
  wrap_f(r, &x);
}

void
xfcos(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_cos(&x, &y));
  wrap_f(r, &x);
}

void
xftan(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_tan(&x, &y));
  wrap_f(r, &x);
}

void
xfasin(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_asin(&x, &y));
  wrap_f(r, &x);
}

void
xfacos(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_acos(&x, &y));
  wrap_f(r, &x);
}

void
xfatan(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_atan(&x, &y));
  wrap_f(r, &x);
}

void
xfatan2(IEEE32 f1, IEEE32 f2, IEEE32 *r)
{
  float32_t x, y, z;
  unwrap_f(&y, &f1);
  unwrap_f(&z, &f2);
  check(fold_real32_atan2(&x, &y, &z));
  wrap_f(r, &x);
}

void
xfexp(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_exp(&x, &y));
  wrap_f(r, &x);
}

void
xflog(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_log(&x, &y));
  wrap_f(r, &x);
}

void
xflog10(IEEE32 f, IEEE32 *r)
{
  float32_t x, y;
  unwrap_f(&y, &f);
  check(fold_real32_log10(&x, &y));
  wrap_f(r, &x);
}

int
xfcmp(IEEE32 f1, IEEE32 f2)
{
  float32_t y, z;
  unwrap_f(&y, &f1);
  unwrap_f(&z, &f2);
  return fold_real32_compare(&y, &z);
}

int
xfisint(IEEE32 f, int *i)
{
  float32_t x, y;
  int64_t k;
  unwrap_f(&x, &f);
  check(fold_int32_from_real32(i, &x));
  k = *i;
  check(fold_real32_from_int64(&y, &k));
  return fold_real32_compare(&x, &y) == FOLD_EQ;
}

/*
 *  Copy a floating-point literal into a null-terminated buffer
 *  so that it may be passed to strtod() et al.  Insert a leading "0x"
 *  after the sign, if requested; also transform a Fortran double-precision
 *  exponent character 'd'/'D'/'q'/'Q' into 'e'.
 */
static void
get_literal(char *buffer, size_t length, const char *s, int n, bool is_hex)
{
  char *p = buffer;
  assert(length > 0);
  while (n > 0 && isspace(*s)) {
    ++s;
    --n;
  }
  if (n > 0 && (*s == '-' || *s == '+') && length > 1) {
    *p++ = *s++;
    --n;
    --length;
  }
  if (is_hex && length > 3) {
    *p++ = '0';
    *p++ = 'x';
    length -= 2;
  }
  while (n > 0 && *s && length > 1) {
    char ch = *s++;
    --n;
    if (!is_hex && (ch == 'd' || ch == 'D' || ch == 'q' || ch == 'Q'))
      ch = 'e';
    *p++ = ch;
    --length;
  }
  *p = '\0';
}

static int
handle_parsing_errors(const char *buffer, const char *end,
		      int errno_capture, bool iszero) {
  if (end != buffer + strlen(buffer))
    return -1; /* format error */
  /* The only error value documented for strtod() is ERANGE,
   * but let's be defensive and produce at least a warning
   * if some other error condition was raised.
   */
  if (errno_capture == ERANGE && iszero)
    return iszero ? -3 /* underflow */ : -2 /* overflow */;
  return errno_capture == 0 ? 0 : -2 /* overflow */;
}

static int
parse_f(const char *s, IEEE32 *f, int n, bool is_hex)
{
  float32_t x;
  char buffer[256], *end;
  int errno_capture;
  get_literal(buffer, sizeof buffer, s, n, is_hex);
  errno = 0;
  x = strtof(buffer, &end);
  errno_capture = errno;
  wrap_f(f, &x);
  return handle_parsing_errors(buffer, end, errno_capture, x == 0);
}

int
atoxf(const char *s, IEEE32 *f, int n)
{
  return parse_f(s, f, n, false);
}

int
hxatoxf(const char *s, IEEE32 *f, int n)
{
  return parse_f(s, f, n, true);
}

/*
 *  64-bit floating-point
 */

void
xdfix(IEEE64 d, INT *i)
{
  float64_t x;
  unwrap_d(&x, d);
  check(fold_int32_from_real64(i, &x));
}

void
xdfixu(IEEE64 d, UINT *u)
{
  float64_t x;
  unwrap_d(&x, d);
  check(fold_uint32_from_real64(u, &x));
}

void
xdfix64(IEEE64 d, DBLINT64 l)
{
  int64_t x;
  float64_t y;
  unwrap_d(&y, d);
  check(fold_int64_from_real64(&x, &y));
  wrap_l(l, &x);
}

void
xdfixu64(IEEE64 d, DBLUINT64 u)
{
  int64_t x;
  float64_t y;
  uint64_t xu;
  unwrap_d(&y, d);
  check(fold_int64_from_real64(&x, &y));
  xu = x;
  wrap_u(u, &xu);
}

void
xdfloat(INT i, IEEE64 d)
{
  float64_t x;
  int64_t li = i;
  check(fold_real64_from_int64(&x, &li));
  wrap_d(d, &x);
}

void
xdfloatu(UINT u, IEEE64 d)
{
  float64_t x;
  int64_t li = u;
  check(fold_real64_from_int64(&x, &li));
  wrap_d(d, &x);
}

void
xdflt64(DBLINT64 l, IEEE64 d)
{
  float64_t x;
  int64_t y;
  unwrap_l(&y, l);
  check(fold_real64_from_int64(&x, &y));
  wrap_d(d, &x);
}

void
xdfltu64(DBLUINT64 u, IEEE64 d)
{
  uint64_t y;
  float64_t x;
  unwrap_u(&y, u);
  check(fold_real64_from_uint64(&x, &y));
  wrap_d(d, &x);
}

void
xdble(IEEE32 f, IEEE64 r)
{
  float64_t x;
  float32_t y;
  unwrap_f(&y, &f);
  check(fold_real64_from_real32(&x, &y));
  wrap_d(r, &x);
}

void
xsngl(IEEE64 d, IEEE32 *r)
{
  float32_t x;
  float64_t y;
  unwrap_d(&y, d);
  check(fold_real32_from_real64(&x, &y));
  wrap_f(r, &x);
}

void
xdadd(IEEE64 d1, IEEE64 d2, IEEE64 r)
{
  float64_t x, y, z;
  unwrap_d(&y, d1);
  unwrap_d(&z, d2);
  check(fold_real64_add(&x, &y, &z));
  wrap_d(r, &x);
}

void
xdsub(IEEE64 d1, IEEE64 d2, IEEE64 r)
{
  float64_t x, y, z;
  unwrap_d(&y, d1);
  unwrap_d(&z, d2);
  check(fold_real64_subtract(&x, &y, &z));
  wrap_d(r, &x);
}

void
xdneg(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_negate(&x, &y));
  wrap_d(r, &x);
}

void
xdmul(IEEE64 d1, IEEE64 d2, IEEE64 r)
{
  float64_t x, y, z;
  unwrap_d(&y, d1);
  unwrap_d(&z, d2);
  check(fold_real64_multiply(&x, &y, &z));
  wrap_d(r, &x);
}

void
xddiv(IEEE64 d1, IEEE64 d2, IEEE64 r)
{
  float64_t x, y, z;
  unwrap_d(&y, d1);
  unwrap_d(&z, d2);
  check(fold_real64_divide(&x, &y, &z));
  wrap_d(r, &x);
}

void
xdrcp(IEEE64 d, IEEE64 r)
{
  float64_t x, one, y;
  int64_t ione = 1;
  unwrap_d(&y, d);
  check(fold_real64_from_int64(&one, &ione));
  check(fold_real64_divide(&x, &one, &y));
  wrap_d(r, &x);
}

void
xdabsv(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_abs(&x, &y));
  wrap_d(r, &x);
}

void
xdsqrt(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_sqrt(&x, &y));
  wrap_d(r, &x);
}

void
xdpow(IEEE64 d1, IEEE64 d2, IEEE64 r)
{
  float64_t x, y, z;
  unwrap_d(&y, d1);
  unwrap_d(&z, d2);
  check(fold_real64_pow(&x, &y, &z));
  wrap_d(r, &x);
}

void
xdsin(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_sin(&x, &y));
  wrap_d(r, &x);
}

void
xdcos(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_cos(&x, &y));
  wrap_d(r, &x);
}

void
xdtan(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_tan(&x, &y));
  wrap_d(r, &x);
}

void
xdasin(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_asin(&x, &y));
  wrap_d(r, &x);
}

void
xdacos(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_acos(&x, &y));
  wrap_d(r, &x);
}

void
xdatan(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_atan(&x, &y));
  wrap_d(r, &x);
}

void
xdatan2(IEEE64 d1, IEEE64 d2, IEEE64 r)
{
  float64_t x, y, z;
  unwrap_d(&y, d1);
  unwrap_d(&z, d2);
  check(fold_real64_atan2(&x, &y, &z));
  wrap_d(r, &x);
}

void
xdexp(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_exp(&x, &y));
  wrap_d(r, &x);
}

void
xdlog(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_log(&x, &y));
  wrap_d(r, &x);
}

void
xdlog10(IEEE64 d, IEEE64 r)
{
  float64_t x, y;
  unwrap_d(&y, d);
  check(fold_real64_log10(&x, &y));
  wrap_d(r, &x);
}

int
xdcmp(IEEE64 d1, IEEE64 d2)
{
  float64_t y, z;
  unwrap_d(&y, d1);
  unwrap_d(&z, d2);
  return fold_real64_compare(&y, &z);
}

int
xdisint(IEEE64 d, int *i)
{
  float64_t x, y;
  int64_t k;
  unwrap_d(&x, d);
  check(fold_int32_from_real64(i, &x));
  k = *i;
  check(fold_real64_from_int64(&y, &k));
  return fold_real64_compare(&x, &y) == FOLD_EQ;
}

static int
parse_d(const char *s, IEEE64 d, int n, bool is_hex)
{
  float64_t x;
  char buffer[256], *end;
  int errno_capture;
  get_literal(buffer, sizeof buffer, s, n, is_hex);
  errno = 0;
  x = strtod(buffer, &end);
  errno_capture = errno;
  wrap_d(d, &x);
  return handle_parsing_errors(buffer, end, errno_capture, x == 0);
}

int
atoxd(const char *s, IEEE64 d, int n)
{
  return parse_d(s, d, n, false);
}

int
hxatoxd(const char *s, IEEE64 d, int n)
{
  return parse_d(s, d, n, true);
}

#ifdef FOLD_LDBL_X87
static void
unwrap_e(float128_t *x, IEEE80 e)
{
  union {
    float128_t x;
    uint32_t i[3];
  } u;
  assert(is_host_little_endian());
  u.i[2] = (uint32_t) e[0] >> 16; /* big end */
  u.i[1] = (e[0] << 16) | ((uint32_t) e[1] >> 16);
  u.i[0] = (e[1] << 16) | ((uint32_t) e[2] >> 16);
  *x = u.x;
}

static void
wrap_e(IEEE80 res, float128_t *x)
{
  union {
    float128_t e;
    uint32_t i[3];
  } u;
  assert(is_host_little_endian());
  u.e = *x;
  res[0] = (u.i[2] << 16) | (u.i[1] >> 16); /* big end */
  res[1] = (u.i[1] << 16) | (u.i[0] >> 16);
  res[2] = u.i[0] << 16;
}

void
xefix(IEEE80 e, INT *i)
{
  float128_t x;
  unwrap_e(&x, e);
  check(fold_int32_from_real128(i, &x));
}

void
xefixu(IEEE80 e, UINT *u)
{
  float128_t x;
  unwrap_e(&x, e);
  check(fold_uint32_from_real128(u, &x));
}

void
xefix64(IEEE80 e, DBLINT64 l)
{
  int64_t x;
  float128_t y;
  unwrap_e(&y, e);
  check(fold_int64_from_real128(&x, &y));
  wrap_l(l, &x);
}

void
xefixu64(IEEE80 e, DBLUINT64 u)
{
  uint64_t x;
  float128_t y;
  unwrap_e(&y, e);
  check(fold_uint64_from_real128(&x, &y));
  wrap_u(u, &x);
}

void
xefloat(INT i, IEEE80 e)
{
  float128_t x;
  int64_t li = i;
  check(fold_real128_from_int64(&x, &li));
  wrap_e(e, &x);
}

void
xefloatu(UINT u, IEEE80 e)
{
  float128_t x;
  int64_t i = u;
  check(fold_real128_from_int64(&x, &i));
  wrap_e(e, &x);
}

void
xeflt64(DBLINT64 l, IEEE80 e)
{
  float128_t x;
  int64_t y;
  unwrap_l(&y, l);
  check(fold_real128_from_int64(&x, &y));
  wrap_e(e, &x);
}

void
xefltu64(DBLUINT64 u, IEEE80 e)
{
  float128_t x;
  uint64_t y;
  unwrap_u(&y, u);
  check(fold_real128_from_uint64(&x, &y));
  wrap_e(e, &x);
}

void
xftoe(IEEE32 f, IEEE80 e)
{
  float128_t x;
  float32_t y;
  unwrap_f(&y, &f);
  check(fold_real128_from_real32(&x, &y));
  wrap_e(e, &x);
}

void
xdtoe(IEEE64 d, IEEE80 e)
{
  float128_t x;
  float64_t y;
  unwrap_d(&y, d);
  check(fold_real128_from_real64(&x, &y));
  wrap_e(e, &x);
}

void
xetof(IEEE80 e, IEEE32 *r)
{
  float32_t x;
  float128_t y;
  unwrap_e(&y, e);
  check(fold_real32_from_real128(&x, &y));
  wrap_f(r, &x);
}

void
xetod(IEEE80 e, IEEE64 d)
{
  float64_t x;
  float128_t y;
  unwrap_e(&y, e);
  check(fold_real64_from_real128(&x, &y));
  wrap_d(d, &x);
}

void
xeadd(IEEE80 e1, IEEE80 e2, IEEE80 r)
{
  float128_t x, y, z;
  unwrap_e(&y, e1);
  unwrap_e(&z, e2);
  check(fold_real128_add(&x, &y, &z));
  wrap_e(r, &x);
}

void
xesub(IEEE80 e1, IEEE80 e2, IEEE80 r)
{
  float128_t x, y, z;
  unwrap_e(&y, e1);
  unwrap_e(&z, e2);
  check(fold_real128_subtract(&x, &y, &z));
  wrap_e(r, &x);
}

void
xeneg(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_negate(&x, &y));
  wrap_e(r, &x);
}

void
xemul(IEEE80 e1, IEEE80 e2, IEEE80 r)
{
  float128_t x, y, z;
  unwrap_e(&y, e1);
  unwrap_e(&z, e2);
  check(fold_real128_multiply(&x, &y, &z));
  wrap_e(r, &x);
}

void
xediv(IEEE80 e1, IEEE80 e2, IEEE80 r)
{
  float128_t x, y, z;
  unwrap_e(&y, e1);
  unwrap_e(&z, e2);
  check(fold_real128_divide(&x, &y, &z));
  wrap_e(r, &x);
}

void
xeabsv(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_abs(&x, &y));
  wrap_e(r, &x);
}

void
xesqrt(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_sqrt(&x, &y));
  wrap_e(r, &x);
}

void
xepow(IEEE80 e1, IEEE80 e2, IEEE80 r)
{
  float128_t x, y, z;
  unwrap_e(&y, e1);
  unwrap_e(&z, e2);
  check(fold_real128_pow(&x, &y, &z));
  wrap_e(r, &x);
}

void
xesin(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_sin(&x, &y));
  wrap_e(r, &x);
}

void
xecos(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_cos(&x, &y));
  wrap_e(r, &x);
}

void
xetan(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_tan(&x, &y));
  wrap_e(r, &x);
}

void
xeasin(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_asin(&x, &y));
  wrap_e(r, &x);
}

void
xeacos(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_acos(&x, &y));
  wrap_e(r, &x);
}

void
xeatan(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_atan(&x, &y));
  wrap_e(r, &x);
}

void
xeatan2(IEEE80 e1, IEEE80 e2, IEEE80 r)
{
  float128_t x, y, z;
  unwrap_e(&y, e1);
  unwrap_e(&z, e2);
  check(fold_real128_atan2(&x, &y, &z));
  wrap_e(r, &x);
}

void
xeexp(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_exp(&x, &y));
  wrap_e(r, &x);
}

void
xelog(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_log(&x, &y));
  wrap_e(r, &x);
}

void
xelog10(IEEE80 e, IEEE80 r)
{
  float128_t x, y;
  unwrap_e(&y, e);
  check(fold_real128_log10(&x, &y));
  wrap_e(r, &x);
}

int
xecmp(IEEE80 e1, IEEE80 e2)
{
  float128_t y, z;
  unwrap_e(&y, e1);
  unwrap_e(&z, e2);
  return fold_real128_compare(&y, &z);
}

static int
parse_e(const char *s, IEEE80 e, int n, bool is_hex)
{
  float128_t x;
  char buffer[256], *end;
  int errno_capture;
  get_literal(buffer, sizeof buffer, s, n, is_hex);
  errno = 0;
  x = strtold(buffer, &end);
  errno_capture = errno;
  wrap_e(e, &x);
  return handle_parsing_errors(buffer, end, errno_capture, x == 0);
}

int
atoxe(const char *s, IEEE80 e, int n)
{
  return parse_e(s, e, n, false);
}

int
hxatoxe(const char *s, IEEE80 e, int n)
{
  return parse_e(s, e, n, true);
}
#endif /* FOLD_LDBL_X87 */

#ifdef FOLD_LDBL_DOUBLEDOUBLE
static void
unwrap_dd(float128_t *x, IEEE6464 dd)
{
  union {
    float128_t x;
    float64_t d[2];
  } u;
  unwrap_d(&u.d[0], dd[0]);
  unwrap_d(&u.d[1], dd[1]);
  *x = u.x;
}

static void
wrap_dd(IEEE6464 res, const float128_t *x)
{
  union {
    float128_t dd;
    float64_t d[2];
  } u;
  u.dd = *x;
  wrap_d(res[0], &u.d[0]);
  wrap_d(res[1], &u.d[1]);
}

void
xddfix(IEEE6464 dd, INT *i)
{
  float128_t y;
  unwrap_dd(&y, dd);
  check(fold_int32_from_real128(i, &y));
}

void
xddfixu(IEEE6464 dd, UINT *u)
{
  float128_t y;
  unwrap_dd(&y, dd);
  check(fold_uint32_from_real128(u, &y));
}

void
xddfix64(IEEE6464 dd, DBLINT64 l)
{
  int64_t x;
  float128_t y;
  unwrap_dd(&y, dd);
  check(fold_int64_from_real128(&x, &y));
  wrap_l(l, &x);
}

void
xddfixu64(IEEE6464 dd, DBLUINT64 u)
{
  uint64_t x;
  float128_t y;
  unwrap_dd(&y, dd);
  check(fold_uint64_from_real128(&x, &y));
  wrap_u(u, &x);
}

void
xddfloat(INT i, IEEE6464 dd)
{
  float128_t x;
  int64_t li = i;
  check(fold_real128_from_int64(&x, &li));
  wrap_dd(dd, &x);
}

void
xddfloatu(UINT u, IEEE6464 dd)
{
  float128_t x;
  int64_t li = u;
  check(fold_real128_from_int64(&x, &li));
  wrap_dd(dd, &x);
}

void
xddflt64(DBLINT64 l, IEEE6464 dd)
{
  float128_t x;
  int64_t y;
  unwrap_l(&y, l);
  check(fold_real128_from_int64(&x, &y));
  wrap_dd(dd, &x);
}

void
xddfltu64(DBLUINT64 u, IEEE6464 dd)
{
  float128_t x;
  uint64_t y;
  unwrap_u(&y, u);
  check(fold_real128_from_uint64(&x, &y));
  wrap_dd(dd, &x);
}

void
xftodd(IEEE32 f, IEEE6464 dd)
{
  float128_t x;
  float32_t y;
  unwrap_f(&y, &f);
  check(fold_real128_from_real32(&x, &y));
  wrap_dd(dd, &x);
}

void
xdtodd(IEEE64 d, IEEE6464 dd)
{
  float128_t x;
  float64_t y;
  unwrap_d(&y, d);
  check(fold_real128_from_real64(&x, &y));
  wrap_dd(dd, &x);
}

void
xddtof(IEEE6464 dd, IEEE32 *r)
{
  float32_t x;
  float128_t y;
  unwrap_dd(&y, dd);
  check(fold_real32_from_real128(&x, &y));
  wrap_f(r, &x);
}

void
xddtod(IEEE6464 dd, IEEE64 d)
{
  float64_t x;
  float128_t y;
  unwrap_dd(&y, dd);
  check(fold_real64_from_real128(&x, &y));
  wrap_d(d, &x);
}

void
xddadd(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r)
{
  float128_t x, y, z;
  unwrap_dd(&y, dd1);
  unwrap_dd(&z, dd2);
  check(fold_real128_add(&x, &y, &z));
  wrap_dd(r, &x);
}

void
xddsub(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r)
{
  float128_t x, y, z;
  unwrap_dd(&y, dd1);
  unwrap_dd(&z, dd2);
  check(fold_real128_subtract(&x, &y, &z));
  wrap_dd(r, &x);
}

void
xddneg(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_negate(&x, &y));
  wrap_dd(r, &x);
}

void
xddmul(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r)
{
  float128_t x, y, z;
  unwrap_dd(&y, dd1);
  unwrap_dd(&z, dd2);
  check(fold_real128_multiply(&x, &y, &z));
  wrap_dd(r, &x);
}

void
xdddiv(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r)
{
  float128_t x, y, z;
  unwrap_dd(&y, dd1);
  unwrap_dd(&z, dd2);
  check(fold_real128_divide(&x, &y, &z));
  wrap_dd(r, &x);
}

void
xddabs(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_abs(&x, &y));
  wrap_dd(r, &x);
}

void
xddsqrt(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_sqrt(&x, &y));
  wrap_dd(r, &x);
}

void
xddpow(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r)
{
  float128_t x, y, z;
  unwrap_dd(&y, dd1);
  unwrap_dd(&z, dd2);
  check(fold_real128_pow(&x, &y, &z));
  wrap_dd(r, &x);
}

void
xddsin(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_sin(&x, &y));
  wrap_dd(r, &x);
}

void
xddcos(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_cos(&x, &y));
  wrap_dd(r, &x);
}

void
xddtan(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_tan(&x, &y));
  wrap_dd(r, &x);
}

void
xddasin(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_asin(&x, &y));
  wrap_dd(r, &x);
}

void
xddacos(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_acos(&x, &y));
  wrap_dd(r, &x);
}

void
xddatan(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_atan(&x, &y));
  wrap_dd(r, &x);
}

void
xddatan2(IEEE6464 dd1, IEEE6464 dd2, IEEE6464 r)
{
  float128_t x, y, z;
  unwrap_dd(&y, dd1);
  unwrap_dd(&z, dd2);
  check(fold_real128_atan2(&x, &y, &z));
  wrap_dd(r, &x);
}

void
xddexp(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_exp(&x, &y));
  wrap_dd(r, &x);
}

void
xddlog(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_log(&x, &y));
  wrap_dd(r, &x);
}

void
xddlog10(IEEE6464 dd, IEEE6464 r)
{
  float128_t x, y;
  unwrap_dd(&y, dd);
  check(fold_real128_log10(&x, &y));
  wrap_dd(r, &x);
}

int
xddcmp(IEEE6464 dd1, IEEE6464 dd2)
{
  float128_t y, z;
  unwrap_dd(&y, dd1);
  unwrap_dd(&z, dd2);
  return fold_real128_compare(&y, &z);
}

static int
parse_dd(const char *s, IEEE6464 dd, int n, bool is_hex)
{
  float128_t x;
  char buffer[256], *end;
  int errno_capture;
  get_literal(buffer, sizeof buffer, s, n, is_hex);
  errno = 0;
  x = strtold(buffer, &end);
  errno_capture = errno;
  wrap_dd(dd, &x);
  return handle_parsing_errors(buffer, end, errno_capture, x == 0);
}

int
atoxdd(const char *s, IEEE6464 dd, int n)
{
  return parse_dd(s, dd, n, false);
}

int
hxatoxdd(const char *s, IEEE6464 dd, int n)
{
  return parse_dd(s, dd, n, true);
}
#endif /* FOLD_LDBL_DOUBLEDOUBLE */

static void
unwrap_q(float128_t *x, IEEE128 q)
{
  union {
    float128_t x;
    uint32_t i[4];
  } u;
  int le = (int) is_host_little_endian() * 3;
  u.i[le ^ 0] = q[0]; /* big end */
  u.i[le ^ 1] = q[1];
  u.i[le ^ 2] = q[2];
  u.i[le ^ 3] = q[3];
  *x = u.x;
}

static void
wrap_q(IEEE128 res, float128_t *x)
{
  union {
    float128_t q;
    uint32_t i[4];
  } u;
  int le = (int) is_host_little_endian() * 3;
  u.q = *x;
  res[0] = u.i[le ^ 0]; /* big end */
  res[1] = u.i[le ^ 1];
  res[2] = u.i[le ^ 2];
  res[3] = u.i[le ^ 3];
}

void
xqfix(IEEE128 q, INT *i)
{
  float128_t y;
  unwrap_q(&y, q);
  check(fold_int32_from_real128(i, &y));
}

void
xqfixu(IEEE128 q, UINT *u)
{
  float128_t y;
  unwrap_q(&y, q);
  check(fold_uint32_from_real128(u, &y));
}

void
xqfix64(IEEE128 q, DBLINT64 l)
{
  int64_t x;
  float128_t y;
  unwrap_q(&y, q);
  check(fold_int64_from_real128(&x, &y));
  wrap_l(l, &x);
}

void
xqfixu64(IEEE128 q, DBLUINT64 u)
{
  uint64_t x;
  float128_t y;
  unwrap_q(&y, q);
  check(fold_uint64_from_real128(&x, &y));
  wrap_u(u, &x);
}

void
xqflt64(DBLINT64 l, IEEE128 q)
{
  float128_t x;
  int64_t y;
  unwrap_l(&y, l);
  check(fold_real128_from_int64(&x, &y));
  wrap_q(q, &x);
}

void
xqfloat(INT i, IEEE128 q)
{
  float128_t x;
  int64_t li = i;
  check(fold_real128_from_int64(&x, &li));
  wrap_q(q, &x);
}

void
xqfloatu(UINT u, IEEE128 q)
{
  float128_t x;
  int64_t li = u;
  check(fold_real128_from_int64(&x, &li));
  wrap_q(q, &x);
}

void
xqfltu64(DBLUINT64 u, IEEE128 q)
{
  float128_t x;
  uint64_t y;
  unwrap_u(&y, u);
  check(fold_real128_from_uint64(&x, &y));
  wrap_q(q, &x);
}

void
xftoq(IEEE32 f, IEEE128 q)
{
  float128_t x;
  float32_t y;
  unwrap_f(&y, &f);
  check(fold_real128_from_real32(&x, &y));
  wrap_q(q, &x);
}

void
xdtoq(IEEE64 d, IEEE128 q)
{
  float128_t x;
  float64_t y;
  unwrap_d(&y, d);
  check(fold_real128_from_real64(&x, &y));
  wrap_q(q, &x);
}

void
xqtof(IEEE128 q, IEEE32 *r)
{
  float32_t x;
  float128_t y;
  unwrap_q(&y, q);
  check(fold_real32_from_real128(&x, &y));
  wrap_f(r, &x);
}

void
xqtod(IEEE128 q, IEEE64 d)
{
  float64_t x;
  float128_t y;
  unwrap_q(&y, q);
  check(fold_real64_from_real128(&x, &y));
  wrap_d(d, &x);
}

void
xqadd(IEEE128 q1, IEEE128 q2, IEEE128 r)
{
  float128_t x, y, z;
  unwrap_q(&y, q1);
  unwrap_q(&z, q2);
  check(fold_real128_add(&x, &y, &z));
  wrap_q(r, &x);
}

void
xqsub(IEEE128 q1, IEEE128 q2, IEEE128 r)
{
  float128_t x, y, z;
  unwrap_q(&y, q1);
  unwrap_q(&z, q2);
  check(fold_real128_subtract(&x, &y, &z));
  wrap_q(r, &x);
}

void
xqneg(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_negate(&x, &y));
  wrap_q(r, &x);
}

void
xqmul(IEEE128 q1, IEEE128 q2, IEEE128 r)
{
  float128_t x, y, z;
  unwrap_q(&y, q1);
  unwrap_q(&z, q2);
  check(fold_real128_multiply(&x, &y, &z));
  wrap_q(r, &x);
}

void
xqdiv(IEEE128 q1, IEEE128 q2, IEEE128 r)
{
  float128_t x, y, z;
  unwrap_q(&y, q1);
  unwrap_q(&z, q2);
  check(fold_real128_divide(&x, &y, &z));
  wrap_q(r, &x);
}

void
xqabsv(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_abs(&x, &y));
  wrap_q(r, &x);
}

void
xqsqrt(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_sqrt(&x, &y));
  wrap_q(r, &x);
}

void
xqpow(IEEE128 q1, IEEE128 q2, IEEE128 r)
{
  float128_t x, y, z;
  unwrap_q(&y, q1);
  unwrap_q(&z, q2);
  check(fold_real128_pow(&x, &y, &z));
  wrap_q(r, &x);
}

void
xqsin(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_sin(&x, &y));
  wrap_q(r, &x);
}

void
xqcos(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_cos(&x, &y));
  wrap_q(r, &x);
}

void
xqtan(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_tan(&x, &y));
  wrap_q(r, &x);
}

void
xqasin(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_asin(&x, &y));
  wrap_q(r, &x);
}

void
xqacos(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_acos(&x, &y));
  wrap_q(r, &x);
}

void
xqatan(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_atan(&x, &y));
  wrap_q(r, &x);
}

void
xqatan2(IEEE128 q1, IEEE128 q2, IEEE128 r)
{
  float128_t x, y, z;
  unwrap_q(&y, q1);
  unwrap_q(&z, q2);
  check(fold_real128_atan2(&x, &y, &z));
  wrap_q(r, &x);
}

void
xqexp(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_exp(&x, &y));
  wrap_q(r, &x);
}

void
xqlog(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_log(&x, &y));
  wrap_q(r, &x);
}

void
xqlog10(IEEE128 q, IEEE128 r)
{
  float128_t x, y;
  unwrap_q(&y, q);
  check(fold_real128_log10(&x, &y));
  wrap_q(r, &x);
}

int
xqcmp(IEEE128 q1, IEEE128 q2)
{
  float128_t y, z;
  unwrap_q(&y, q1);
  unwrap_q(&z, q2);
  return fold_real128_compare(&y, &z);
}

#ifdef TARGET_SUPPORTS_QUADFP
int
xqisint(IEEE128 q, int *i)
{
  float128_t x, y;
  int64_t k;
  unwrap_q(&x, q);
  check(fold_int32_from_real128(i, &x));
  if (i == NULL)
    return 0;

  k = *i;
  check(fold_real128_from_int64(&y, &k));
  return fold_real128_compare(&x, &y) == FOLD_EQ;
}

void
xmqtoq(float128_t mq, IEEE128 q)
{
  wrap_q(q, &mq);
}
#endif

static int
parse_q(const char *s, IEEE128 q, int n, bool is_hex)
{
  float128_t x;
  char buffer[256], *end;
  int errno_capture;
  get_literal(buffer, sizeof buffer, s, n, is_hex);
  errno = 0;
  x = strtold(buffer, &end);
  errno_capture = errno;
  wrap_q(q, &x);
  return handle_parsing_errors(buffer, end, errno_capture, x == 0);
}

int
atoxq(const char *s, IEEE128 q, int n)
{
  return parse_q(s, q, n, false);
}

int
hxatoxq(const char *s, IEEE128 q, int n)
{
  return parse_q(s, q, n, true);
}

/*
 *  Miscellaneous, possibly unused
 */

INT
xudiv(UINT n, UINT d, UINT *r)
{
  if (d == 0)
    return -1;
  *r = n / d;
  return 0;
}

INT
xumod(UINT n, UINT d, UINT *r)
{
  if (d == 0)
    return -1;
  *r = n % d;
  return 0;
}

/* Does an unsigned comparison, but declared in scutil with signed args. */
INT
xucmp(INT sa, INT sb)
{
  uint32_t a = sa, b = sb;
  return a < b ? -1 : a > b;
}

/*
 *  Literal integer scanning routines
 */

/* Utility subroutine for atoxi()/atoxi64(). Ensure that the literal
 * is null-terminated and actually contains some digits.  Recognize
 * any leading sign character.
 */
static const char *
get_int_literal(const char *s, int n, char *buffer, size_t buffer_length,
                bool *is_negative)
{
  *is_negative = false;

  while (n > 0 && isspace(*s)) {
    --n, ++s;
  }
  if (n > 0) {
    if (*s == '+') {
      --n, ++s; /* ignore '+' */
    } else if (*s == '-') {
      *is_negative = true;
      --n, ++s;
    }
  }
  if (n < 1)
    return NULL;
  if (!memchr(s, '\0', n)) {
    /* copy to ensure null termination */
    if (n > buffer_length + 1)
      return NULL;
    memcpy(buffer, s, n);
    buffer[n] = '\0';
    return buffer;
  }
  return s;
}

static int
generic_atoxi(uint64_t *v, bool *is_negative,
              const char *s, int n, int base)
{
  char buffer[64], *end;
  if (!(s = get_int_literal(s, n, buffer, sizeof buffer, is_negative)))
    return -1; /* syntax or size error */
  errno = 0;
  *v = strtoull(s, &end, abs(base));
  if (errno == ERANGE) {
    /* overflow */
    return base < 0 ? -3 : -2;
  }
  if (errno != 0 || *end != '\0')
    return -1; /* syntax */
  return 0;
}

int
atoxi(const char *s, INT *i, int n, int base)
{
  uint64_t v;
  bool is_negative = false;
  int err = generic_atoxi(&v, &is_negative, s, n, base);
  if (!err && v > INT32_MAX) {
    if (base < 0)
      err = -3;
    else if (is_negative || v > UINT32_MAX)
      err = -2;
  }
  *i = is_negative ? -v : v;
  return err;
}

int
atoxi64(const char *s, DBLINT64 i, int n, int base)
{
  uint64_t v;
  int64_t sv;
  bool is_negative = false;
  int err = generic_atoxi(&v, &is_negative, s, n, base);
  if (!err && (int64_t) v < 0) {
    if (base < 0)
      err = -3;
    else if (is_negative)
      err = -2;
  }
  sv = is_negative ? -v : v;
  wrap_l(i, &sv);
  return err;
}

/*
 *  These two variants force a signed interpretation of decimals.
 */

int
atosi32(const char *s, INT *i, int n, int base)
{
  return atoxi(s, i, n, base == 10 ? -10 : base);
}

int
atosi64(const char *s, DBLINT64 i, int n, int base)
{
  return atoxi64(s, i, n, base == 10 ? -10 : base);
}

/*
 *  Format a floating-point constant.  The format character determines
 *  the size/type of the constant.
 */
void
cprintf(char *buffer, const char *format, INT *val)
{
  size_t off = strspn(format, " %-0123456789.");
  float128_t x;
  float64_t d;
  float32_t f;
  const char *p = format + off;
  char nfmt[128];
  IEEE32 fv;

  buffer[0] = '\0';

  switch (*p) {
  case 'q':
  case 'L':
#ifdef FOLD_LDBL_X87
    unwrap_e(&x, val);
#elif defined FOLD_LDBL_DOUBLEDOUBLE
    unwrap_dd(&x, (IEEE64*) val);
#elif defined FOLD_LDBL_128BIT
    unwrap_q(&x, val);
#else
    return;
#endif
    ++p;
    break;
  case 'l':
    unwrap_d(&d, val);
    x = d;
    ++p;
    break;
  default:
    fv = (intptr_t) val;
    unwrap_f(&f, &fv);
    x = f;
  }
  if (off <= sizeof nfmt - 3) {
    memcpy(nfmt, format, off);
    strcpy(nfmt + off, "LE");
    sprintf(buffer, nfmt, x);
    if (*p == 'd' || *p == 'q') {
      char *E = strchr(buffer, 'E');
      if (E != NULL)
        *E = 'D';
    }
  }
}

void
xcfpow(IEEE32 r1, IEEE32 i1, IEEE32 r2, IEEE32 i2, IEEE32 *rr, IEEE32 *ir)
{
  FLOAT_COMPLEX_TYPE x, y, z;
  float32_t rx, ix, ry, iy, rz, iz;
  unwrap_f(&rx, &r1);
  unwrap_f(&ix, &i1);
  unwrap_f(&ry, &r2);
  unwrap_f(&iy, &i2);
  if (rx == 0.0 && ix == 0.0 && ry == 0.0 && iy == 0.0) {
    rz = 1.0;
    iz = 0.0;
  } else {
    x = FLOAT_COMPLEX_CREATE(rx, ix);
    y = FLOAT_COMPLEX_CREATE(ry, iy);
    check(fold_complex32_pow(&z, &x, &y));
    rz = crealf(z);
    iz = cimagf(z);
  }
  wrap_f(rr, &rz);
  wrap_f(ir, &iz);
}

void
xcdpow(IEEE64 r1, IEEE64 i1, IEEE64 r2, IEEE64 i2, IEEE64 rr, IEEE64 ir)
{
  DOUBLE_COMPLEX_TYPE x, y, z;
  float64_t rx, ix, ry, iy, rz, iz;
  unwrap_d(&rx, r1);
  unwrap_d(&ix, i1);
  unwrap_d(&ry, r2);
  unwrap_d(&iy, i2);
  if (rx == 0.0 && ix == 0.0 && ry == 0.0 && iy == 0.0) {
    rz = 1.0;
    iz = 0.0;
  } else {
    x = DOUBLE_COMPLEX_CREATE(rx, ix);
    y = DOUBLE_COMPLEX_CREATE(ry, iy);
    check(fold_complex64_pow(&z, &x, &y));
    rz = creal(z);
    iz = cimag(z);
  }
  wrap_d(rr, &rz);
  wrap_d(ir, &iz);
}

void
xcqpow(IEEE128 r1, IEEE128 i1, IEEE128 r2, IEEE128 i2, IEEE128 rr, IEEE128 ir)
{
  LONG_DOUBLE_COMPLEX_TYPE x, y, z;
  float128_t rx, ix, ry, iy, rz, iz;
  unwrap_q(&rx, r1);
  unwrap_q(&ix, i1);
  unwrap_q(&ry, r2);
  unwrap_q(&iy, i2);
  if (rx == 0.0 && ix == 0.0 && ry == 0.0 && iy == 0.0) {
    rz = 1.0;
    iz = 0.0;
  } else {
    x = LONG_DOUBLE_COMPLEX_CREATE(rx, ix);
    y = LONG_DOUBLE_COMPLEX_CREATE(ry, iy);
    check(fold_complex128_pow(&z, &x, &y));
    rz = creall(z);
    iz = cimagl(z);
  }
  wrap_q(rr, &rz);
  wrap_q(ir, &iz);
}
