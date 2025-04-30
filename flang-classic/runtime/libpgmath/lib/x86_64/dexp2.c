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

#define USE_SPLITEXP
#define USE_SCALEDOUBLE_1
#define USE_SCALEDOUBLE_2
#define USE_ZERO_WITH_FLAGS
#define USE_INFINITY_WITH_FLAGS
#define USE_HANDLE_ERROR
#include "libm_inlines_amd.h"
#undef USE_ZERO_WITH_FLAGS
#undef USE_SPLITEXP
#undef USE_SCALEDOUBLE_1
#undef USE_SCALEDOUBLE_2
#undef USE_INFINITY_WITH_FLAGS
#undef USE_HANDLE_ERROR

#include "libm_errno_amd.h"

/* Deal with errno for out-of-range result */
static inline double
retval_errno_erange_overflow(double x __attribute__((unused)))
{
  return infinity_with_flags(AMD_F_OVERFLOW | AMD_F_INEXACT);
}

static inline double
retval_errno_erange_underflow(double x __attribute__((unused)))
{
  return zero_with_flags(AMD_F_UNDERFLOW | AMD_F_INEXACT);
}

double FN_PROTOTYPE(mth_i_dexp2)(double x)
{
  static const double max_exp2_arg = 1024.0,  /* 0x4090000000000000 */
      min_exp2_arg = -1074.0,                 /* 0xc090c80000000000 */
      log2 = 6.931471805599453094178e-01,     /* 0x3fe62e42fefa39ef */
      log2_lead = 6.93147167563438415527E-01, /* 0x3fe62e42f8000000 */
      log2_tail = 1.29965068938898869640E-08, /* 0x3e4be8e7bcd5e4f1 */
      one_by_32_lead = 0.03125;

  double y, z1, z2, z, hx, tx, y1, y2;
  int m;
  __UINT8_T ux, ax;

  /*
    Computation of exp2(x).

    We compute the values m, z1, and z2 such that
    exp2(x) = 2**m * (z1 + z2),  where exp2(x) is 2**x.

    Computations needed in order to obtain m, z1, and z2
    involve three steps.

    First, we reduce the argument x to the form
    x = n/32 + remainder,
    where n has the value of an integer and |remainder| <= 1/64.
    The value of n = x * 32 rounded to the nearest integer and
    the remainder = x - n/32.

    Second, we approximate exp2(r1 + r2) - 1 where r1 is the leading
    part of the remainder and r2 is the trailing part of the remainder.

    Third, we reconstruct exp2(x) so that
    exp2(x) = 2**m * (z1 + z2).
  */

  GET_BITS_DP64(x, ux);
  ax = ux & (~SIGNBIT_DP64);

  if (ax >= 0x4090000000000000) /* abs(x) >= 1024.0 */
  {
    if (ax >= 0x7ff0000000000000) {
      /* x is either NaN or infinity */
      if (ux & MANTBITS_DP64)
        /* x is NaN */
        return x + x; /* Raise invalid if it is a signalling NaN */
      else if (ux & SIGNBIT_DP64)
        /* x is negative infinity; return 0.0 with no flags. */
        return 0.0;
      else
        /* x is positive infinity */
        return x;
    }
    if (x > max_exp2_arg)
      /* Return +infinity with overflow flag */
      return retval_errno_erange_overflow(x);
    else if (x < min_exp2_arg)
      /* x is negative. Return +zero with underflow and inexact flags */
      return retval_errno_erange_underflow(x);
  }

  /* Handle small arguments separately */
  if (ax < 0x3fb7154764ee6c2f) /* abs(x) < 1/(16*log2) */
  {
    if (ax < 0x3c00000000000000) /* abs(x) < 2^(-63) */
      return 1.0 + x;            /* Raises inexact if x is non-zero */
    else {
      /* Split x into hx (head) and tx (tail). */
      __UINT8_T u;
      hx = x;
      GET_BITS_DP64(hx, u);
      u &= 0xfffffffff8000000;
      PUT_BITS_DP64(u, hx);
      tx = x - hx;
      /* Carefully multiply x by log2. y1 is the most significant
         part of the result, and y2 the least significant part */
      y1 = x * log2_lead;
      y2 = (((hx * log2_lead - y1) + hx * log2_tail) + tx * log2_lead) +
           tx * log2_tail;

      y = y1 + y2;
      z = (9.99564649780173690e-1 +
           (1.61251249355268050e-5 +
            (2.37986978239838493e-2 + 2.68724774856111190e-7 * y) * y) *
               y) /
          (9.99564649780173692e-1 +
           (-4.99766199765151309e-1 +
            (1.070876894098586184e-1 +
             (-1.189773642681502232e-2 + 5.9480622371960190616e-4 * y) * y) *
                y) *
               y);
      z = ((z * y1) + (z * y2)) + 1.0;
    }
  } else {
    /* Find m, z1 and z2 such that exp2(x) = 2**m * (z1 + z2) */

    splitexp(x, log2, 32.0, one_by_32_lead, 0.0, &m, &z1, &z2);

    /* Scale (z1 + z2) by 2.0**m */
    if (m > EMIN_DP64 && m < EMAX_DP64)
      z = scaleDouble_1((z1 + z2), m);
    else
      z = scaleDouble_2((z1 + z2), m);
  }
  return z;
}
