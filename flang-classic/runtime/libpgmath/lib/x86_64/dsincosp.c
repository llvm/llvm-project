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

#define USE_REMAINDER_PIBY2_INLINE
#define USE_NAN_WITH_FLAGS
#define USE_VAL_WITH_FLAGS
#define USE_HANDLE_ERROR
#include "libm_inlines_amd.h"
#undef USE_NAN_WITH_FLAGS
#undef USE_VAL_WITH_FLAGS
#undef USE_HANDLE_ERROR
#undef USE_REMAINDER_PIBY2_INLINE

/* sin(x) approximation valid on the interval [-pi/4,pi/4]. */
static inline double
sin_piby4(double x, double xx)
{
  /* Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
                          = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
                          = x * f(w)
     where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
     We use a minimax approximation of (f(w) - 1) / w
     because this produces an expansion in even powers of x.
     If xx (the tail of x) is non-zero, we add a correction
     term g(x,xx) = (1-x*x/2)*xx to the result, where g(x,xx)
     is an approximation to cos(x)*sin(xx) valid because
     xx is tiny relative to x.
  */
  static const double c1 = -0.166666666666666646259241729,
                      c2 = 0.833333333333095043065222816e-2,
                      c3 = -0.19841269836761125688538679e-3,
                      c4 = 0.275573161037288022676895908448e-5,
                      c5 = -0.25051132068021699772257377197e-7,
                      c6 = 0.159181443044859136852668200e-9;
  double x2, x3, r;
  x2 = x * x;
  x3 = x2 * x;
  r = (c2 + x2 * (c3 + x2 * (c4 + x2 * (c5 + x2 * c6))));
  if (xx == 0.0)
    return x + x3 * (c1 + x2 * r);
  else
    return x - ((x2 * (0.5 * xx - x3 * r) - xx) - x3 * c1);
}

/* cos(x) approximation valid on the interval [-pi/4,pi/4]. */
static inline double
cos_piby4(double x, double xx)
{
  /* Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
                          = f(w)
     where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
     We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
     because this produces an expansion in even powers of x.
     If xx (the tail of x) is non-zero, we subtract a correction
     term g(x,xx) = x*xx to the result, where g(x,xx)
     is an approximation to sin(x)*sin(xx) valid because
     xx is tiny relative to x.
  */
  double r, x2, t;
  static const double c1 = 0.41666666666666665390037e-1,
                      c2 = -0.13888888888887398280412e-2,
                      c3 = 0.248015872987670414957399e-4,
                      c4 = -0.275573172723441909470836e-6,
                      c5 = 0.208761463822329611076335e-8,
                      c6 = -0.113826398067944859590880e-10;

  x2 = x * x;
  r = 0.5 * x2;
  t = 1.0 - r;
  return t +
         ((((1.0 - t) - r) - x * xx) +
          x2 * x2 *
              (c1 + x2 * (c2 + x2 * (c3 + x2 * (c4 + x2 * (c5 + x2 * c6))))));
}

void FN_PROTOTYPE(mth_dsincos)(double x, double *s, double *c)
{
  double r, rr;
  int region, xneg;

  __UINT8_T ux, ax;
  GET_BITS_DP64(x, ux);
  ax = (ux & ~SIGNBIT_DP64);
  if (ax <= 0x3fe921fb54442d18) /* abs(x) <= pi/4 */
  {
    if (ax < 0x3f20000000000000) /* abs(x) < 2.0^(-13) */
    {
      if (ax < 0x3e40000000000000) /* abs(x) < 2.0^(-27) */
      {
        if (ax == 0x0000000000000000) {
          *s = x;
          *c = 1.0;
        } else {
          *s = x;
          *c = val_with_flags(1.0, AMD_F_INEXACT);
        }
      } else {
        *s = x - x * x * x * 0.166666666666666666;
        *c = 1.0 - x * x * 0.5;
      }
    } else {
      *s = sin_piby4(x, 0.0);
      *c = cos_piby4(x, 0.0);
    }
    return;
  } else if ((ux & EXPBITS_DP64) == EXPBITS_DP64) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_DP64)
      /* x is NaN */
      *s = *c = x + x; /* Raise invalid if it is a signalling NaN */
    else
      /* x is infinity. Return a NaN */
      *s = *c = nan_with_flags(AMD_F_INVALID);
    return;
  }

  xneg = (ax != ux);

  if (xneg)
    x = -x;

  if (x < 5.0e5) {
    /* For these size arguments we can just carefully subtract the
       appropriate multiple of pi/2, using extra precision where
       x is close to an exact multiple of pi/2 */
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
    /* How many pi/2 is x a multiple of? */
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
      npi2 = (int)(x * twobypi + 0.5);
    /* Subtract the multiple from x to get an extra-precision remainder */
    rhead = x - npi2 * piby2_1;
    rtail = npi2 * piby2_1tail;
    GET_BITS_DP64(rhead, uy);
    expdiff = xexp - ((uy & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
    if (expdiff > 15) {
      /* The remainder is pretty small compared with x, which
         implies that x is a near multiple of pi/2
         (x matches the multiple to at least 15 bits) */
      t = rhead;
      rtail = npi2 * piby2_2;
      rhead = t - rtail;
      rtail = npi2 * piby2_2tail - ((t - rhead) - rtail);
      if (expdiff > 48) {
        /* x matches a pi/2 multiple to at least 48 bits */
        t = rhead;
        rtail = npi2 * piby2_3;
        rhead = t - rtail;
        rtail = npi2 * piby2_3tail - ((t - rhead) - rtail);
      }
    }
    r = rhead - rtail;
    rr = (rhead - r) - rtail;
    region = npi2 & 3;
  } else {
    /* Reduce x into range [-pi/4,pi/4] */
    __remainder_piby2_inline(x, &r, &rr, &region);
  }

  if (xneg) {
    switch (region) {
    default:
    case 0:
      *s = -sin_piby4(r, rr);
      *c = cos_piby4(r, rr);
      break;
    case 1:
      *s = -cos_piby4(r, rr);
      *c = -sin_piby4(r, rr);
      break;
    case 2:
      *s = sin_piby4(r, rr);
      *c = -cos_piby4(r, rr);
      break;
    case 3:
      *s = cos_piby4(r, rr);
      *c = sin_piby4(r, rr);
      break;
    }
  } else {
    switch (region) {
    default:
    case 0:
      *s = sin_piby4(r, rr);
      *c = cos_piby4(r, rr);
      break;
    case 1:
      *s = cos_piby4(r, rr);
      *c = -sin_piby4(r, rr);
      break;
    case 2:
      *s = -sin_piby4(r, rr);
      *c = -cos_piby4(r, rr);
      break;
    case 3:
      *s = -cos_piby4(r, rr);
      *c = sin_piby4(r, rr);
      break;
    }
  }
  return;
}

double FN_PROTOTYPE(mth_i_dsin)(double x)
{
  double r, rr;
  int region, xneg;

  __UINT8_T ux, ax;
  GET_BITS_DP64(x, ux);
  ax = (ux & ~SIGNBIT_DP64);
  if (ax <= 0x3fe921fb54442d18) /* abs(x) <= pi/4 */
  {
    if (ax < 0x3f20000000000000) /* abs(x) < 2.0^(-13) */
    {
      if (ax < 0x3e40000000000000) /* abs(x) < 2.0^(-27) */
      {
        if (ax == 0x0000000000000000)
          return x;
        else
          return val_with_flags(x, AMD_F_INEXACT);
      } else
        return x - x * x * x * 0.166666666666666666;
    } else
      return sin_piby4(x, 0.0);
  } else if ((ux & EXPBITS_DP64) == EXPBITS_DP64) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_DP64)
      /* x is NaN */
      return x + x; /* Raise invalid if it is a signalling NaN */
    else
      /* x is infinity. Return a NaN */
      return nan_with_flags(AMD_F_INVALID);
  }
  xneg = (ax != ux);

  if (xneg)
    x = -x;

  /* Reduce x into range [-pi/4,pi/4] */
  if (x < 5.0e5) {
    /* For these size arguments we can just carefully subtract the
       appropriate multiple of pi/2, using extra precision where
       x is close to an exact multiple of pi/2 */
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
    /* How many pi/2 is x a multiple of? */
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
      npi2 = (int)(x * twobypi + 0.5);
    /* Subtract the multiple from x to get an extra-precision remainder */
    rhead = x - npi2 * piby2_1;
    rtail = npi2 * piby2_1tail;
    GET_BITS_DP64(rhead, uy);
    expdiff = xexp - ((uy & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
    if (expdiff > 15) {
      /* The remainder is pretty small compared with x, which
         implies that x is a near multiple of pi/2
         (x matches the multiple to at least 15 bits) */
      t = rhead;
      rtail = npi2 * piby2_2;
      rhead = t - rtail;
      rtail = npi2 * piby2_2tail - ((t - rhead) - rtail);
      if (expdiff > 48) {
        /* x matches a pi/2 multiple to at least 48 bits */
        t = rhead;
        rtail = npi2 * piby2_3;
        rhead = t - rtail;
        rtail = npi2 * piby2_3tail - ((t - rhead) - rtail);
      }
    }
    r = rhead - rtail;
    rr = (rhead - r) - rtail;
    region = npi2 & 3;
  } else {
    __remainder_piby2_inline(x, &r, &rr, &region);
  }

  if (xneg) {
    switch (region) {
    default:
    case 0:
      return -sin_piby4(r, rr);
    case 1:
      return -cos_piby4(r, rr);
    case 2:
      return sin_piby4(r, rr);
    case 3:
      return cos_piby4(r, rr);
    }
  } else {
    switch (region) {
    default:
    case 0:
      return sin_piby4(r, rr);
    case 1:
      return cos_piby4(r, rr);
    case 2:
      return -sin_piby4(r, rr);
    case 3:
      return -cos_piby4(r, rr);
    }
  }
}

double FN_PROTOTYPE(mth_i_dcos)(double x)
{
  double r, rr;
  int region, xneg;

  __UINT8_T ux, ax;
  GET_BITS_DP64(x, ux);
  ax = (ux & ~SIGNBIT_DP64);
  if (ax <= 0x3fe921fb54442d18) /* abs(x) <= pi/4 */
  {
    if (ax < 0x3f20000000000000) /* abs(x) < 2.0^(-13) */
    {
      if (ax < 0x3e40000000000000) /* abs(x) < 2.0^(-27) */
      {
        if (ax == 0x0000000000000000) /* abs(x) = 0.0 */
          return 1.0;
        else
          return val_with_flags(1.0, AMD_F_INEXACT);
      } else
        return 1.0 - x * x * 0.5;
    } else
      return cos_piby4(x, 0.0);
  } else if ((ux & EXPBITS_DP64) == EXPBITS_DP64) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_DP64)
      /* x is NaN */
      return x + x; /* Raise invalid if it is a signalling NaN */
    else
      /* x is infinity. Return a NaN */
      return nan_with_flags(AMD_F_INVALID);
  }
  xneg = (ax != ux);

  if (xneg)
    x = -x;

  if (x < 5.0e5) {
    /* For these size arguments we can just carefully subtract the
       appropriate multiple of pi/2, using extra precision where
       x is close to an exact multiple of pi/2 */
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
    /* How many pi/2 is x a multiple of? */
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
      npi2 = (int)(x * twobypi + 0.5);
    /* Subtract the multiple from x to get an extra-precision remainder */
    rhead = x - npi2 * piby2_1;
    rtail = npi2 * piby2_1tail;
    GET_BITS_DP64(rhead, uy);
    expdiff = xexp - ((uy & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
    if (expdiff > 15) {
      /* The remainder is pretty small compared with x, which
         implies that x is a near multiple of pi/2
         (x matches the multiple to at least 15 bits) */
      t = rhead;
      rtail = npi2 * piby2_2;
      rhead = t - rtail;
      rtail = npi2 * piby2_2tail - ((t - rhead) - rtail);
      if (expdiff > 48) {
        /* x matches a pi/2 multiple to at least 48 bits */
        t = rhead;
        rtail = npi2 * piby2_3;
        rhead = t - rtail;
        rtail = npi2 * piby2_3tail - ((t - rhead) - rtail);
      }
    }
    r = rhead - rtail;
    rr = (rhead - r) - rtail;
    region = npi2 & 3;
  } else {
    /* Reduce x into range [-pi/4,pi/4] */
    __remainder_piby2_inline(x, &r, &rr, &region);
  }

  switch (region) {
  default:
  case 0:
    return cos_piby4(r, rr);
  case 1:
    return -sin_piby4(r, rr);
  case 2:
    return -cos_piby4(r, rr);
  case 3:
    return sin_piby4(r, rr);
  }
}
