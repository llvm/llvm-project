/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* These routines are included in linux and osx.
   I've implemented them so we can also claim support on Windows.
   Plus, we can standardize our support of F2003 ieee_exceptions
   and ieee_arithmetic modules across all platforms

   Plus, Cray is asking for them.  Not sure they know they are in Linux
   Hope to find out in mid-2009

   The OSX implementation and our implementation are similar.  Linux
   gnu does not use x87 in 64 bits, and doesn't seem to use mxcsr in
   32 bits.

     - Brent
*/

#include "float128.h"

float __mth_i_around(float x);
float __mth_i_remainder(float x, float y);
double __mth_i_dround(double x);
double __mth_i_dremainder(double x, double y);
#ifdef TARGET_SUPPORTS_QUADFP
float128_t __mth_i_qround(float128_t x);
float128_t __mth_i_qremainder(float128_t x, float128_t y);
long double scalbnl(long double x, int i);
#endif
int __fenv_fegetzerodenorm(void);

#ifdef TARGET_SUPPORTS_QUADFP
long double
__nearbyintl(long double x)
{
  return __mth_i_qround(x);
}
#endif

double
__nearbyint(double x)
{
  return __mth_i_dround(x);
}

float
__nearbyintf(float x)
{
  return __mth_i_around(x);
}

#ifdef TARGET_SUPPORTS_QUADFP
long double
__remainderl(long double x, long double y)
{
  return __mth_i_qremainder(x, y);
}
#endif

double
__remainder(double x, double y)
{
  return __mth_i_dremainder(x, y);
}

float
__remainderf(float x, float y)
{
  return __mth_i_remainder(x, y);
}

#ifdef TARGET_SUPPORTS_QUADFP
static inline int
isnan(unsigned int h, unsigned int j, unsigned int k, unsigned int l)
{
    return (((h & 0x7fff0000) == 0x7fff0000) &&
           ((l | k | j | (h & 0xffff)) != 0));
}
#endif

#define _MAXLONGDOUBLE (1.18973149535723176508575932662800702E+4932L)
#define IX_SIZE 4 /* ix[] has 4 element and 128 bits */
#define IZ_SIZE 2 /* iz[] has 4 element and 128 bits */
#define GET_SIGN 31
#define GET_EXP1 16
#define GET_EXP2 23
#define IS_ZERO(h, j, k, l) (((h) & 0x7fffffff) == 0) && (((l) | (k) | (j)) == 0)
#define IS_MIN_NORM(h, j, k, l, s) (((h) & 0x7fffffff) == 0x00010000) && (((l) | (k) | (j)) == 0) && (s)
#define IS_MAX_NORM(h, j, k, l ,s) (((h) & 0x7fffffff) == 0x7ffeffff) && (((~((l) & 0xffffffff)) | \
                           (~((k) & 0xffffffff)) | (~((j) & 0xffffffff))) == 0) && (!(s))
#ifdef TARGET_SUPPORTS_QUADFP
long double
__nextafterl(long double x, long double y)
{
  long double ex;
  unsigned int ix[IX_SIZE], ixh, ixj, ixk, ixl, iyh, iyj, iyk, iyl;
  unsigned long long iz[IZ_SIZE], izh, izl;
  int idflag, subflag;

  ex = y;
  iyl = *((int *)(&(ex)));
  iyk = *((int *)(&(ex)) + 1);
  iyj = *((int *)(&(ex)) + 2);
  iyh = *((int *)(&(ex)) + 3);

  ex = x;
  ixl = *((int *)(&(ex)));
  ixk = *((int *)(&(ex)) + 1);
  ixj = *((int *)(&(ex)) + 2);
  ixh = *((int *)(&(ex)) + 3);

  izl = *((long long *)(&(ex)));
  izh = *((long long *)(&(ex)) + 1);

  /* y is nan, return y */
  if (isnan(iyh, iyj, iyk, iyl))
    return y;

  /* x is nan, return x */
  if (isnan(ixh, ixj, ixk, ixl))
    return x;

  /* x infinity, return -_MAXLONGDOUBLE, +_MAXLONGDOUBLE, or x */
  if ((ixh & 0x7fff0000) == 0x7fff0000) {
    if ((iyh & 0x7fff0000) == 0x7fff0000) {
      if ((ixh >> GET_SIGN) ^ (iyh >> GET_SIGN))
        return (int)(ixh >> GET_SIGN) ? -_MAXLONGDOUBLE : _MAXLONGDOUBLE;
      return x;
    }
    if (!(ixh & 0x80000000))
      /* +inf, any number */
      return _MAXLONGDOUBLE;
    /* -inf, any number */
    return -_MAXLONGDOUBLE;
  }

  if (x == y)
    return x;
  subflag = (x > y);

  if (IS_ZERO(ixh, ixj, ixk ,ixl)) {
    idflag = __fenv_fegetzerodenorm();
    ix[3] = ((unsigned int) subflag << GET_SIGN) | ((unsigned int) idflag << GET_EXP1);
    ix[2] = 0x00000000;
    ix[1] = 0x00000000;
    ix[0] = (unsigned int) !idflag;
    ex = *((long double *)ix);
    return x + ex; // underflow here
  } else {
    if (ixh & 0x80000000)
      subflag = !subflag;
    if (IS_MIN_NORM(ixh, ixj, ixk, ixl, subflag)) {
      /* In this special case, setup a value to subtract
         so we cause exceptions to occur properly
      */
      idflag = __fenv_fegetzerodenorm();
      ix[3] = (ixh & 0x80000000);
      ix[3] = ix[3] | ((unsigned int) idflag << GET_EXP2);
      ix[2] = 0x00000000;
      ix[1] = 0x00000000;
      ix[0] = (unsigned int) !idflag;
      ex = *((long double *)ix);
      return x - ex; // possible underflow here
    } else if (IS_MAX_NORM(ixh, ixj, ixk, ixl, subflag)) {
      ix[3] = (ixh & 0xffca0000);
      ix[2] = 0;
      ix[1] = 0;
      ix[0] = 0;
      ex = *((long double *)ix);
      return x + ex; // overflow to infinity
    } else {
      /* This is the normal case */
        if (subflag) {
          if (izl == 0) {
            iz[0] = izl - 1;
            iz[1] = izh - 1;
          } else {
            iz[0] = izl - 1;
            iz[1] = izh;
          }
        } else {
          iz[0] = izl + 1;
          if (iz[0] == 0) {
            iz[1] = izh + 1;
          } else {
            iz[1] = izh;
        }
      }
    }
  }
  ex = *((long double *)iz);
  return ex;
}
#endif

#define _MAXDOUBLE (1.7976931348623157e+308)

double
__nextafter(double x, double y)
{
  double ex;
  unsigned int ix[2], ixh, ixl, iyh, iyl;
  int idflag, subflag;

  ex = y;
  iyl = *((int *)(&(ex)));
  iyh = *((int *)(&(ex)) + 1);

  ex = x;
  ixl = *((int *)(&(ex)));
  ixh = *((int *)(&(ex)) + 1);

  /* y is nan, return y */
  if (((iyh & 0x7ff00000) == 0x7ff00000) &&
      ((iyl != 0) || (iyh & 0x0fffff) != 0))
    return y;

  /* x is nan, return x */
  if (((ixh & 0x7ff00000) == 0x7ff00000) &&
      ((ixl != 0) || (ixh & 0x0fffff) != 0))
    return y;

  /* x infinity, return -_MAXDOUBLE, +_MAXDOUBLE, or x */
  if ((ixh & 0x7ff00000) == 0x7ff00000) {
    if ((iyh & 0x7ff00000) == 0x7ff00000) {
      if (!(ixh & 0x80000000)) {
        if (iyh & 0x80000000)
          /* inf, -inf */
          return _MAXDOUBLE;
      } else if (!(iyh & 0x80000000))
        /* -inf, +inf */
        return -_MAXDOUBLE;
      return x;
    }
    if (!(ixh & 0x80000000))
      /* +inf, any number */
      return _MAXDOUBLE;
    /* -inf, any number */
    return -_MAXDOUBLE;
  }

  if (x == y)
    return x;
  subflag = (x > y);

  if (((ixh & 0x7fffffff) == 0) && (ixl == 0)) {
    idflag = __fenv_fegetzerodenorm();
    if (idflag) {
      if (subflag) {
        ix[1] = 0x80100000;
        ix[0] = 0x00000000;
      } else {
        ix[1] = 0x00100000;
        ix[0] = 0x00000000;
      }
    } else {
      if (subflag) {
        ix[1] = 0x80000000;
        ix[0] = 0x00000001;
      } else {
        ix[1] = 0x00000000;
        ix[0] = 0x00000001;
      }
      ex = *((double *)ix);
      return x + ex; // underflow here
    }
  } else {
    if (ixh & 0x80000000)
      subflag = !subflag;
    if (((ixh & 0x7fffffff) == 0x00100000) && (ixl == 0) && (subflag)) {
      /* In this special case, setup a value to subtract
         so we cause exceptions to occur properly
      */
      idflag = __fenv_fegetzerodenorm();
      ix[1] = (ixh & 0x80000000);
      if (idflag) {
        ix[1] = ix[1] | 0x00800000;
        ix[0] = 0x00000000;
      } else {
        ix[0] = 0x00000001;
      }
      ex = *((double *)ix);
      return x - ex; // possible underflow here

    } else if (((ixh & 0x7fffffff) == 0x7fefffff) && (ixl == 0xffffffff) &&
               (!subflag)) {
      ix[1] = (ixh & 0xfca00000);
      ix[0] = 0;
      ex = *((double *)ix);
      return x + ex; // overflow to infinity
    } else {
      /* This is the normal case */
      if (subflag) {
        if (ixl == 0) {
          ix[0] = ixl - 1;
          ix[1] = ixh - 1;
        } else {
          ix[0] = ixl - 1;
          ix[1] = ixh;
        }
      } else {
        ix[0] = ixl + 1;
        if (ix[0] == 0) {
          ix[1] = ixh + 1;
        } else {
          ix[1] = ixh;
        }
      }
    }
  }
  ex = *((double *)ix);
  return ex;
}

#define _MAXFLOAT (3.40282347e+38F)

float
__nextafterf(float x, float y)
{
  float ex;
  unsigned int ix, iy, iz;
  int idflag, subflag;

  ex = y;
  iy = *((int *)(&(ex)));
  ex = x;
  ix = *((int *)(&(ex)));

  /* y is nan, return y */
  if (((iy & 0x7f800000) == 0x7f800000) && ((iy & 0x7fffff) != 0))
    return y;

  /* x is nan  return x */
  if (((ix & 0x7f800000) == 0x7f800000) && ((ix & 0x7fffff) != 0))
    return x;

  /* x infinity, return -_MAXFLOAT, +_MAXFLOAT, or x */
  if ((ix & 0x7f800000) == 0x7f800000) {
    if ((iy & 0x7f800000) == 0x7f800000) {
      if (!(ix & 0x80000000)) {
        if (iy & 0x80000000)
          /* inf, -inf */
          return _MAXFLOAT;
      } else if (!(iy & 0x80000000))
        /* -inf, +inf */
        return -_MAXFLOAT;
      return x;
    }
    if (!(ix & 0x80000000))
      /* +inf, any number */
      return _MAXFLOAT;
    /* -inf, any number */
    return -_MAXFLOAT;
  }

  if (x == y)
    return x;
  subflag = (x > y);

  if ((ix & 0x7fffffff) == 0) {
    idflag = __fenv_fegetzerodenorm();
    if (idflag) {
      if (subflag)
        iz = 0x80800000;
      else
        iz = 0x00800000;
    } else {
      if (subflag)
        iz = 0x80000001;
      else
        iz = 0x00000001;
      ex = *((float *)&iz);
      return x + ex; // underflow here
    }
  } else {
    if (ix & 0x80000000)
      subflag = !subflag;
    if (((ix & 0x7fffffff) == 0x00800000) && (subflag)) {
      idflag = __fenv_fegetzerodenorm();
      iz = (ix & 0x80000000);
      if (idflag)
        iz = iz | 0x00800000;
      else
        iz = iz | 0x00000001;
      ex = *((float *)&iz);
      return x - ex; // possible underflow here
    } else if (((ix & 0x7fffffff) == 0x7f7fffff) && (!subflag)) {
      iz = (ix & 0xf3000000);
      ex = *((float *)&iz);
      return x + ex; // overflow to infinity
    } else {
      if (subflag)
        iz = ix - 1;
      else
        iz = ix + 1;
    }
  }
  ex = *((float *)&iz);
  return ex;
}

#ifdef TARGET_SUPPORTS_QUADFP
long double
__scalbnl(long double x, int i)
{
  return scalbnl(x,i);
}
#endif

double
__scalbn(double x, int i)
{
  /* Do the scaling in three parts.  Should allow for full range of
     scaling, but still will generate underflow/overflow where appropriate
  */
  double ex, fx;
  int ix[2], iy[2], iz[2];
  ix[1] = i;
  ix[0] = 0;
  iy[1] = 0;
  iy[0] = 0;
  iz[1] = 0;
  iz[0] = 0;
  if (i > 1000) {
    ix[1] = 1000;
    iy[1] = ((i - ix[1]) < 1000) ? i - ix[1] : 1000;
    iz[1] = ((i - ix[1] - iy[1]) < 1000) ? i - ix[1] - iy[1] : 1000;
  } else if (i < -1000) {
    ix[1] = -1000;
    iy[1] = ((i - ix[1]) > -1000) ? i - ix[1] : -1000;
    iz[1] = ((i - ix[1] - iy[1]) > -1000) ? i - ix[1] - iy[1] : -1000;
  }
  ix[1] = ix[1] + 1023;
  ix[1] = (ix[1] << 20);
  ex = *((double *)ix);
  fx = x * ex;
  if (iy[1] != 0) {
    iy[1] = iy[1] + 1023;
    iy[1] = (iy[1] << 20);
    ex = *((double *)iy);
    fx = fx * ex;
  }
  if (iz[1] != 0) {
    iz[1] = iz[1] + 1023;
    iz[1] = (iz[1] << 20);
    ex = *((double *)iz);
    fx = fx * ex;
  }
  return fx;
}

float
__scalbnf(float x, int i)
{
  /* Do the scaling in three parts.  Should allow for full range of
     scaling, but still will generate underflow/overflow where appropriate
  */
  float ex, fx;
  int ix, iy, iz;
  ix = i;
  iy = 0;
  iz = 0;
  if (i > 120) {
    ix = 120;
    iy = ((i - ix) < 120) ? i - ix : 120;
    iz = ((i - ix - iy) < 120) ? i - ix - iy : 120;
  } else if (i < -120) {
    ix = -120;
    iy = ((i - ix) > -120) ? i - ix : -120;
    iz = ((i - ix - iy) > -120) ? i - ix - iy : -120;
  }
  ix = ix + 127;
  ix = (ix << 23);
  ex = *((float *)&ix);
  fx = x * ex;
  if (iy != 0) {
    iy = iy + 127;
    iy = (iy << 23);
    ex = *((float *)&iy);
    fx = fx * ex;
  }
  if (iz != 0) {
    iz = iz + 127;
    iz = (iz << 23);
    ex = *((float *)&iz);
    fx = fx * ex;
  }
  return fx;
}

