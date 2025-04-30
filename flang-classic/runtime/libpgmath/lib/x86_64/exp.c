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

#define USE_SPLITEXPF
#define USE_SCALEFLOAT_1
#define USE_SCALEFLOAT_2
#define USE_ZEROF_WITH_FLAGS
#define USE_INFINITYF_WITH_FLAGS
#define USE_NANF_WITH_FLAGS
#define USE_HANDLE_ERRORF
#include "libm_inlines_amd.h"
#undef USE_SPLITEXPF
#undef USE_SCALEFLOAT_1
#undef USE_SCALEFLOAT_2
#undef USE_ZEROF_WITH_FLAGS
#undef USE_INFINITYF_WITH_FLAGS
#undef USE_NANF_WITH_FLAGS
#undef USE_HANDLE_ERRORF

#include "libm_errno_amd.h"

/* Deal with errno for out-of-range result */
static inline float
retval_errno_erange_overflow(float x __attribute__((unused)))
{
  return infinityf_with_flags(AMD_F_OVERFLOW | AMD_F_INEXACT);
}

static inline float
retval_errno_erange_underflow(float x __attribute__((unused)))
{
  return zerof_with_flags(AMD_F_UNDERFLOW | AMD_F_INEXACT);
}

float FN_PROTOTYPE(mth_i_exp)(float x)
{
  static const float max_exp_arg = 8.8722839355E+01F, /* 0x42B17218 */
      min_exp_arg = -1.0327893066E+02F,               /* 0xC2CE8ED0 */
      thirtytwo_by_log2 = 4.6166240692E+01F,          /* 0x4238AA3B */
      log2_by_32_lead = 2.1659851074E-02F,            /* 0x3CB17000 */
      log2_by_32_tail = 9.9831822808E-07F;            /* 0x3585FDF4 */

  float z1, z2, z;
  int m;
  unsigned int ux, ax;

  /*
    Computation of exp(x).

    We compute the values m, z1, and z2 such that
    exp(x) = 2**m * (z1 + z2),  where
    exp(x) is the natural exponential of x.

    Computations needed in order to obtain m, z1, and z2
    involve three steps.

    First, we reduce the argument x to the form
    x = n * log2/32 + remainder,
    where n has the value of an integer and |remainder| <= log2/64.
    The value of n = x * 32/log2 rounded to the nearest integer and
    the remainder = x - n*log2/32.

    Second, we approximate exp(r1 + r2) - 1 where r1 is the leading
    part of the remainder and r2 is the trailing part of the remainder.

    Third, we reconstruct the exponential of x so that
    exp(x) = 2**m * (z1 + z2).
  */

  GET_BITS_SP32(x, ux);
  if ((ux << 1) == 0) {
    /* Quick exit if arg == +/- 0.0 */
    return 1.0;
  }
  ax = ux & (~SIGNBIT_SP32);

  if (ax >= 0x42B17218) /* abs(x) >= 88.7... */
  {
    if (ax >= 0x7f800000) {
      /* x is either NaN or infinity */
      if (ux & MANTBITS_SP32)
        /* x is NaN */
        return x + x; /* Raise invalid if it is a signalling NaN */
      else if (ux & SIGNBIT_SP32)
        /* x is negative infinity; return 0.0 with no flags */
        return 0.0;
      else
        /* x is positive infinity */
        return x;
    }
    if (x > max_exp_arg)
      /* Return +infinity with overflow flag */
      return retval_errno_erange_overflow(x);
    else if (x < min_exp_arg)
      /* x is negative. Return +zero with underflow and inexact flags */
      return retval_errno_erange_underflow(x);
  }

  /* Handle small arguments separately */
  if (ax < 0x3c800000) /* abs(x) < 1/64 */
  {
    if (ax < 0x32800000) /* abs(x) < 2^(-26) */
      return 1.0F + x;   /* Raises inexact if x is non-zero */
    else
      z = (((((((1.0F / 5040) * x + 1.0F / 720) * x + 1.0F / 120) * x +
              1.0F / 24) *
                 x +
             1.0F / 6) *
                x +
            1.0F / 2) *
               x +
           1.0F) *
              x +
          1.0F;
  } else {
    /* Find m and z such that exp(x) = 2**m * (z1 + z2) */

    splitexpf(x, 1.0F, thirtytwo_by_log2, log2_by_32_lead, log2_by_32_tail, &m,
              &z1, &z2);

    /* Scale (z1 + z2) by 2.0**m */

    if (m >= EMIN_SP32 && m <= EMAX_SP32)
      z = scaleFloat_1((float)(z1 + z2), m);
    else
      z = scaleFloat_2((float)(z1 + z2), m);
  }
  return z;
}
