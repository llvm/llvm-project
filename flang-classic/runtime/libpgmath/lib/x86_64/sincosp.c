/* ============================================================
Copyright (c) 2002-2015 Advanced Micro Devices, Inc.

All rights reserved.

Redistribution and  use in source and binary  forms, with or
without  modification,  are   permitted  provided  that  the
following conditions are met:

+ Redistributions  of source  code  must  retain  the  above
  copyright  notice,   this  list  of   conditions  and  the
  following disclaimer.

+ Redistributions  in binary  form must reproduce  the above
  copyright  notice,   this  list  of   conditions  and  the
  following  disclaimer in  the  documentation and/or  other
  materials provided with the distribution.

+ Neither the  name of Advanced Micro Devices,  Inc. nor the
  names  of  its contributors  may  be  used  to endorse  or
  promote  products  derived   from  this  software  without
  specific prior written permission.

THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
CONTRIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
INC.  OR CONTRIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
(INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
LIABILITY,  WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
(INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
POSSIBILITY OF SUCH DAMAGE.

It is  licensee's responsibility  to comply with  any export
regulations applicable in licensee's jurisdiction.
============================================================ */

#include "libm_amd.h"
#include "libm_util_amd.h"

#define USE_REMAINDER_PIBY2F_INLINE
#define USE_VALF_WITH_FLAGS
#define USE_NANF_WITH_FLAGS
#define USE_HANDLE_ERRORF
#include "libm_inlines_amd.h"
#undef USE_VALF_WITH_FLAGS
#undef USE_NANF_WITH_FLAGS
#undef USE_REMAINDER_PIBY2F_INLINE
#undef USE_HANDLE_ERRORF

/* sin(x) approximation valid on the interval [-pi/4,pi/4]. */
static inline double
sinf_piby4(double x)
{
  /* Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
                          = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
                          = x * f(w)
     where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
     We use a minimax approximation of (f(w) - 1) / w
     because this produces an expansion in even powers of x.
  */
  double x2;
  static const double c1 = -0.166666666638608441788607926e0,
                      c2 = 0.833333187633086262120839299e-2,
                      c3 = -0.198400874359527693921333720e-3,
                      c4 = 0.272500015145584081596826911e-5;

  x2 = x * x;
  return (x + x * x2 * (c1 + x2 * (c2 + x2 * (c3 + x2 * c4))));
}

/* cos(x) approximation valid on the interval [-pi/4,pi/4]. */
static inline double
cosf_piby4(double x)
{
  /* Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
                          = f(w)
     where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
     We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
     because this produces an expansion in even powers of x.
  */
  double x2;
  static const double c1 = 0.41666666664325175238031e-1,
                      c2 = -0.13888887673175665567647e-2,
                      c3 = 0.24800600878112441958053e-4,
                      c4 = -0.27301013343179832472841e-6;

  x2 = x * x;
  return (1.0 - 0.5 * x2 + (x2 * x2 * (c1 + x2 * (c2 + x2 * (c3 + x2 * c4)))));
}

void FN_PROTOTYPE(mth_sincos)(float x, float *s, float *c)
{
  double r, dx;
  int region, xneg;

  __UINT8_T ux, ax;

  dx = x;

  GET_BITS_DP64(dx, ux);
  ax = (ux & ~SIGNBIT_DP64);

  if (ax <= 0x3fe921fb54442d18) /* abs(x) <= pi/4 */
  {
    if (ax < 0x3f80000000000000) /* abs(x) < 2.0^(-7) */
    {
      if (ax < 0x3f20000000000000) /* abs(x) < 2.0^(-13) */
      {
        if (ax == 0x0000000000000000) {
          *s = x;
          *c = 1.0F;
        } else {
          *s = valf_with_flags(x, AMD_F_INEXACT);
          *c = valf_with_flags(1.0F, AMD_F_INEXACT);
        }
      } else {
        *s = (float)(dx - dx * dx * dx * 0.166666666666666666);
        *c = (float)(1.0 - dx * dx * 0.5);
      }
    } else {
      *s = (float)sinf_piby4(x);
      *c = (float)cosf_piby4(x);
    }
    return;
  } else if ((ux & EXPBITS_DP64) == EXPBITS_DP64) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_DP64)
      /* x is NaN */
      *s = *c = x + x; /* Raise invalid if it is a signalling NaN */
    else
      /* x is infinity. Return a NaN */
      *s = *c = nanf_with_flags(AMD_F_INVALID);
    return;
  }

  xneg = (int)(ux >> 63);

  if (xneg)
    dx = -dx;

  if (dx < 5.0e5) {
    /* For these size arguments we can just carefully subtract the
       appropriate multiple of pi/2, using extra precision where
       dx is close to an exact multiple of pi/2 */
    static const double twobypi =
                            6.36619772367581382433e-01, /* 0x3fe45f306dc9c883 */
        piby2_1 = 1.57079632673412561417e+00,           /* 0x3ff921fb54400000 */
        piby2_1tail = 6.07710050650619224932e-11,       /* 0x3dd0b4611a626331 */
        piby2_2 = 6.07710050630396597660e-11,           /* 0x3dd0b4611a600000 */
        piby2_2tail = 2.02226624879595063154e-21,       /* 0x3ba3198a2e037073 */
        piby2_3 = 2.02226624871116645580e-21,           /* 0x3ba3198a2e000000 */
        piby2_3tail = 8.47842766036889956997e-32;       /* 0x397b839a252049c1 */
    double t, rhead, rtail;
    int npi2;
    __UINT8_T uy, xexp, expdiff;
    xexp = ax >> EXPSHIFTBITS_DP64;
    /* How many pi/2 is dx a multiple of? */
    if (ax <= 0x400f6a7a2955385e) /* 5pi/4 */
    {
      if (ax <= 0x4002d97c7f3321d2) /* 3pi/4 */
        npi2 = 1;
      else
        npi2 = 2;
    } else if (ax <= 0x401c463abeccb2bb) /* 9pi/4 */
    {
      if (ax <= 0x4015fdbbe9bba775) /* 7pi/4 */
        npi2 = 3;
      else
        npi2 = 4;
    } else
      npi2 = (int)(dx * twobypi + 0.5);
    /* Subtract the multiple from dx to get an extra-precision remainder */
    rhead = dx - npi2 * piby2_1;
    rtail = npi2 * piby2_1tail;
    GET_BITS_DP64(rhead, uy);
    expdiff = xexp - ((uy & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
    if (expdiff > 15) {
      /* The remainder is pretty small compared with dx, which
         implies that dx is a near multiple of pi/2
         (dx matches the multiple to at least 15 bits) */
      t = rhead;
      rtail = npi2 * piby2_2;
      rhead = t - rtail;
      rtail = npi2 * piby2_2tail - ((t - rhead) - rtail);
      if (expdiff > 48) {
        /* dx matches a pi/2 multiple to at least 48 bits */
        t = rhead;
        rtail = npi2 * piby2_3;
        rhead = t - rtail;
        rtail = npi2 * piby2_3tail - ((t - rhead) - rtail);
      }
    }
    r = rhead - rtail;
    region = npi2 & 3;
  } else {
    /* Reduce abs(dx) into range [-pi/4,pi/4] */
    __remainder_piby2f_inline(ax, &r, &region);
  }

  if (xneg) {
    switch (region) {
    default:
    case 0:
      *s = (float)-sinf_piby4(r);
      *c = (float)cosf_piby4(r);
      break;
    case 1:
      *s = (float)-cosf_piby4(r);
      *c = (float)-sinf_piby4(r);
      break;
    case 2:
      *s = (float)sinf_piby4(r);
      *c = (float)-cosf_piby4(r);
      break;
    case 3:
      *s = (float)cosf_piby4(r);
      *c = (float)sinf_piby4(r);
      break;
    }
  } else {
    switch (region) {
    default:
    case 0:
      *s = (float)sinf_piby4(r);
      *c = (float)cosf_piby4(r);
      break;
    case 1:
      *s = (float)cosf_piby4(r);
      *c = (float)-sinf_piby4(r);
      break;
    case 2:
      *s = (float)-sinf_piby4(r);
      *c = (float)-cosf_piby4(r);
      break;
    case 3:
      *s = (float)-cosf_piby4(r);
      *c = (float)sinf_piby4(r);
      break;
    }
  }
}

float FN_PROTOTYPE(mth_i_sin)(float x)
{
  double r, dx;
  int region, xneg;

  __UINT8_T ux, ax;

  dx = x;

  GET_BITS_DP64(dx, ux);
  ax = (ux & ~SIGNBIT_DP64);

  if (ax <= 0x3fe921fb54442d18) /* abs(x) <= pi/4 */
  {
    if (ax < 0x3f80000000000000) /* abs(x) < 2.0^(-7) */
    {
      if (ax < 0x3f20000000000000) /* abs(x) < 2.0^(-13) */
      {
        if (ax == 0x0000000000000000)
          return x;
        else
          return valf_with_flags(x, AMD_F_INEXACT);
      } else
        return (float)(dx - dx * dx * dx * 0.166666666666666666);
    } else
      return (float)sinf_piby4(dx);
  } else if ((ux & EXPBITS_DP64) == EXPBITS_DP64) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_DP64)
      /* x is NaN */
      return x + x; /* Raise invalid if it is a signalling NaN */
    else
      /* x is infinity. Return a NaN */
      return nanf_with_flags(AMD_F_INVALID);
  }

  xneg = (int)(ux >> 63);

  if (xneg)
    dx = -dx;

  if (dx < 5.0e5) {
    /* For these size arguments we can just carefully subtract the
       appropriate multiple of pi/2, using extra precision where
       dx is close to an exact multiple of pi/2 */
    static const double twobypi =
                            6.36619772367581382433e-01, /* 0x3fe45f306dc9c883 */
        piby2_1 = 1.57079632673412561417e+00,           /* 0x3ff921fb54400000 */
        piby2_1tail = 6.07710050650619224932e-11,       /* 0x3dd0b4611a626331 */
        piby2_2 = 6.07710050630396597660e-11,           /* 0x3dd0b4611a600000 */
        piby2_2tail = 2.02226624879595063154e-21,       /* 0x3ba3198a2e037073 */
        piby2_3 = 2.02226624871116645580e-21,           /* 0x3ba3198a2e000000 */
        piby2_3tail = 8.47842766036889956997e-32;       /* 0x397b839a252049c1 */
    double t, rhead, rtail;
    int npi2;
    __UINT8_T uy, xexp, expdiff;
    xexp = ax >> EXPSHIFTBITS_DP64;
    /* How many pi/2 is dx a multiple of? */
    if (ax <= 0x400f6a7a2955385e) /* 5pi/4 */
    {
      if (ax <= 0x4002d97c7f3321d2) /* 3pi/4 */
        npi2 = 1;
      else
        npi2 = 2;
    } else if (ax <= 0x401c463abeccb2bb) /* 9pi/4 */
    {
      if (ax <= 0x4015fdbbe9bba775) /* 7pi/4 */
        npi2 = 3;
      else
        npi2 = 4;
    } else
      npi2 = (int)(dx * twobypi + 0.5);
    /* Subtract the multiple from dx to get an extra-precision remainder */
    rhead = dx - npi2 * piby2_1;
    rtail = npi2 * piby2_1tail;
    GET_BITS_DP64(rhead, uy);
    expdiff = xexp - ((uy & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
    if (expdiff > 15) {
      /* The remainder is pretty small compared with dx, which
         implies that dx is a near multiple of pi/2
         (dx matches the multiple to at least 15 bits) */
      t = rhead;
      rtail = npi2 * piby2_2;
      rhead = t - rtail;
      rtail = npi2 * piby2_2tail - ((t - rhead) - rtail);
      if (expdiff > 48) {
        /* dx matches a pi/2 multiple to at least 48 bits */
        t = rhead;
        rtail = npi2 * piby2_3;
        rhead = t - rtail;
        rtail = npi2 * piby2_3tail - ((t - rhead) - rtail);
      }
    }
    r = rhead - rtail;
    region = npi2 & 3;
  } else {
    /* Reduce abs(x) into range [-pi/4,pi/4] */
    __remainder_piby2f_inline(ax, &r, &region);
  }

  if (xneg) {
    switch (region) {
    default:
    case 0:
      return (float)-sinf_piby4(r);
    case 1:
      return (float)-cosf_piby4(r);
    case 2:
      return (float)sinf_piby4(r);
    case 3:
      return (float)cosf_piby4(r);
    }
  } else {
    switch (region) {
    default:
    case 0:
      return (float)sinf_piby4(r);
    case 1:
      return (float)cosf_piby4(r);
    case 2:
      return (float)-sinf_piby4(r);
    case 3:
      return (float)-cosf_piby4(r);
    }
  }
}

/* This is the way cosf should be done, but it sometimes runs half
   as fast as it ought to, with gcc on Hammer */

float FN_PROTOTYPE(mth_i_cos)(float x)
{
  double r, dx;
  int region, xneg;

  __UINT8_T ux, ax;

  dx = x;

  GET_BITS_DP64(dx, ux);
  ax = (ux & ~SIGNBIT_DP64);

  if (ax <= 0x3fe921fb54442d18) /* abs(x) <= pi/4 */
  {
    if (ax < 0x3f80000000000000) /* abs(x) < 2.0^(-7) */
    {
      if (ax < 0x3f20000000000000) /* abs(x) < 2.0^(-13) */
      {
        if (ax == 0x0000000000000000)
          return 1.0F;
        else
          return valf_with_flags(1.0F, AMD_F_INEXACT);
      } else
        return (float)(1.0 - dx * dx * 0.5);
    } else
      return cosf_piby4(dx);
  } else if ((ux & EXPBITS_DP64) == EXPBITS_DP64) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_DP64)
      /* x is NaN */
      return x + x; /* Raise invalid if it is a signalling NaN */
    else
      /* x is infinity. Return a NaN */
      return nanf_with_flags(AMD_F_INVALID);
  }

  xneg = (ux >> 63);

  if (xneg)
    dx = -dx;

  if (dx < 5.0e5) {
    /* For these size arguments we can just carefully subtract the
       appropriate multiple of pi/2, using extra precision where
       dx is close to an exact multiple of pi/2 */
    static const double twobypi =
                            6.36619772367581382433e-01, /* 0x3fe45f306dc9c883 */
        piby2_1 = 1.57079632673412561417e+00,           /* 0x3ff921fb54400000 */
        piby2_1tail = 6.07710050650619224932e-11,       /* 0x3dd0b4611a626331 */
        piby2_2 = 6.07710050630396597660e-11,           /* 0x3dd0b4611a600000 */
        piby2_2tail = 2.02226624879595063154e-21,       /* 0x3ba3198a2e037073 */
        piby2_3 = 2.02226624871116645580e-21,           /* 0x3ba3198a2e000000 */
        piby2_3tail = 8.47842766036889956997e-32;       /* 0x397b839a252049c1 */
    double t, rhead, rtail;
    int npi2;
    __UINT8_T uy, xexp, expdiff;
    xexp = ax >> EXPSHIFTBITS_DP64;
    /* How many pi/2 is dx a multiple of? */
    if (ax <= 0x400f6a7a2955385e) /* 5pi/4 */
    {
      if (ax <= 0x4002d97c7f3321d2) /* 3pi/4 */
        npi2 = 1;
      else
        npi2 = 2;
    } else if (ax <= 0x401c463abeccb2bb) /* 9pi/4 */
    {
      if (ax <= 0x4015fdbbe9bba775) /* 7pi/4 */
        npi2 = 3;
      else
        npi2 = 4;
    } else
      npi2 = (int)(dx * twobypi + 0.5);
    /* Subtract the multiple from dx to get an extra-precision remainder */
    rhead = dx - npi2 * piby2_1;
    rtail = npi2 * piby2_1tail;
    GET_BITS_DP64(rhead, uy);
    expdiff = xexp - ((uy & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
    if (expdiff > 15) {
      /* The remainder is pretty small compared with dx, which
         implies that dx is a near multiple of pi/2
         (dx matches the multiple to at least 15 bits) */
      t = rhead;
      rtail = npi2 * piby2_2;
      rhead = t - rtail;
      rtail = npi2 * piby2_2tail - ((t - rhead) - rtail);
      if (expdiff > 48) {
        /* dx matches a pi/2 multiple to at least 48 bits */
        t = rhead;
        rtail = npi2 * piby2_3;
        rhead = t - rtail;
        rtail = npi2 * piby2_3tail - ((t - rhead) - rtail);
      }
    }
    r = rhead - rtail;
    region = npi2 & 3;
  } else {
    /* Reduce abs(x) into range [-pi/4,pi/4] */
    __remainder_piby2f_inline(ax, &r, &region);
  }

  switch (region) {
  default:
  case 0:
    return (float)cosf_piby4(r);
  case 1:
    return (float)-sinf_piby4(r);
  case 2:
    return (float)-cosf_piby4(r);
  case 3:
    return (float)sinf_piby4(r);
  }
}
