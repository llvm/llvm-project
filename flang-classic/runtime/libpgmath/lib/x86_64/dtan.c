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

/* tan(x + xx) approximation valid on the interval [-pi/4,pi/4].
   If recip is true return -1/tan(x + xx) instead. */
static inline double
tan_piby4(double x, double xx, int recip)
{
  double r, t1, t2, xl;
  int transform = 0;
  static const double piby4_lead =
                          7.85398163397448278999e-01, /* 0x3fe921fb54442d18 */
      piby4_tail = 3.06161699786838240164e-17;        /* 0x3c81a62633145c06 */

  /* In order to maintain relative precision transform using the identity:
     tan(pi/4-x) = (1-tan(x))/(1+tan(x)) for arguments close to pi/4.
     Similarly use tan(x-pi/4) = (tan(x)-1)/(tan(x)+1) close to -pi/4. */

  if (x > 0.68) {
    transform = 1;
    x = piby4_lead - x;
    xl = piby4_tail - xx;
    x += xl;
    xx = 0.0;
  } else if (x < -0.68) {
    transform = -1;
    x = piby4_lead + x;
    xl = piby4_tail + xx;
    x += xl;
    xx = 0.0;
  }

  /* Core Remez [2,3] approximation to tan(x+xx) on the
     interval [0,0.68]. */

  r = x * x + 2.0 * x * xx;
  t1 = x;
  t2 = xx +
       x * r * (0.372379159759792203640806338901e0 +
                (-0.229345080057565662883358588111e-1 +
                 0.224044448537022097264602535574e-3 * r) *
                    r) /
           (0.111713747927937668539901657944e1 +
            (-0.515658515729031149329237816945e0 +
             (0.260656620398645407524064091208e-1 -
              0.232371494088563558304549252913e-3 * r) *
                 r) *
                r);

  /* Reconstruct tan(x) in the transformed case. */

  if (transform) {
    double t;
    t = t1 + t2;
    if (recip)
      return transform * (2 * t / (t - 1) - 1.0);
    else
      return transform * (1.0 - 2 * t / (1 + t));
  }

  if (recip) {
    /* Compute -1.0/(t1 + t2) accurately */
    double trec, trec_top, z1, z2, t;
    __UINT8_T u;
    t = t1 + t2;
    GET_BITS_DP64(t, u);
    u &= 0xffffffff00000000;
    PUT_BITS_DP64(u, z1);
    z2 = t2 - (z1 - t1);
    trec = -1.0 / t;
    GET_BITS_DP64(trec, u);
    u &= 0xffffffff00000000;
    PUT_BITS_DP64(u, trec_top);
    return trec_top + trec * ((1.0 + trec_top * z1) + trec_top * z2);

  } else
    return t1 + t2;
}

double FN_PROTOTYPE(mth_i_dtan)(double x)
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
      } else {
        return x + x * x * x * 0.333333333333333333;
      }
    } else
      return tan_piby4(x, 0.0, 0);
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

  if (xneg)
    return -tan_piby4(r, rr, region & 1);
  else
    return tan_piby4(r, rr, region & 1);
}
