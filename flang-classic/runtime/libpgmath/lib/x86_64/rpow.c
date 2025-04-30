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

#define USE_ZEROF_WITH_FLAGS
#define USE_INFINITYF_WITH_FLAGS
#define USE_NANF_WITH_FLAGS
#define USE_HANDLE_ERRORF
#include "libm_inlines_amd.h"
#undef USE_ZEROF_WITH_FLAGS
#undef USE_INFINITYF_WITH_FLAGS
#undef USE_NANF_WITH_FLAGS
#undef USE_HANDLE_ERRORF

#include "libm_errno_amd.h"

/* Deal with errno for out-of-range result */
static inline float
retval_errno_erange_overflow(float x __attribute__((unused)), float y __attribute__((unused)), int sign)
{
  if (sign == 1)
    return infinityf_with_flags(AMD_F_OVERFLOW);
  else /* sign == -1 */
    return -infinityf_with_flags(AMD_F_OVERFLOW);
}

static inline float
retval_errno_erange_underflow(float x __attribute__((unused)), float y __attribute__((unused)), int sign)
{
  if (sign == 1)
    return zerof_with_flags(AMD_F_UNDERFLOW | AMD_F_INEXACT);
  else /* sign == -1 */
    return -zerof_with_flags(AMD_F_UNDERFLOW | AMD_F_INEXACT);
}

/* Deal with errno for out-of-range arguments */
static inline float
retval_errno_edom(float x __attribute__((unused)), float y __attribute__((unused)), int type)
{
  if (type == 1)
    return infinityf_with_flags(AMD_F_DIVBYZERO);
  else if (type == 2)
    return -infinityf_with_flags(AMD_F_DIVBYZERO);
  else /* type == 3 */
    return nanf_with_flags(AMD_F_INVALID);
}

static volatile int dummy;

float FN_PROTOTYPE(mth_i_rpowr)(float x, float y)
{
  unsigned int ux, ax, uy, ay, mask;
  int yexp, inty, xpos, ypos, negateres;
  double dx, dy, dw, dlog2, dr;

  /* Largest float, stored as a double */
  const double large = 3.40282346638528859812e+38; /* 0x47efffffe0000000 */

  /* Smallest float, stored as a double */
  const double tiny = 1.40129846432481707092e-45; /* 0x36a0000000000000 */

  GET_BITS_SP32(x, ux);
  ax = ux & (~SIGNBIT_SP32);
  xpos = ax == ux;
  GET_BITS_SP32(y, uy);
  ay = uy & (~SIGNBIT_SP32);
  ypos = ay == uy;

  if (ux == 0x3f800000) {
    /* x = +1.0. Return +1.0 for all y, even NaN,
       raising invalid only if y is a signalling NaN */
    if (y + 1.0F == 2.0F)
      dummy = 1;
    return 1.0F;
  } else if (uy == 0x3fc00000) {
    /* y is 1.5. */
    /* Return x * sqrt(x), even if x is infinity or NaN */
    return (x * sqrt(x));
  } else if (uy == 0x3f000000) {
    /* y is 0.5. */
    /* Return sqrt(x), even if x is infinity or NaN */
    return (sqrt(x));
  } else if (uy == 0x3e800000) {
    /* y is 0.25. */
    /* Return sqrt(sqrt(x)), even if x is infinity or NaN */
    return (sqrt(sqrt(x)));
  } else if (ay == 0) {
    /* y is zero. */
    /* y is zero. Return 1.0, even if x is infinity or NaN,
       raising invalid only if x is a signalling NaN */
    if (x + 1.0F == 2.0F)
      dummy = 1;
    return 1.0F;
  } else if (((ax & EXPBITS_SP32) == EXPBITS_SP32) && (ax & MANTBITS_SP32)) {
    /* x is NaN. Return NaN, with invalid exception if it's
       a signalling NaN. */
    return x + x;
  } else if (((ay & EXPBITS_SP32) == EXPBITS_SP32) && (ay & MANTBITS_SP32)) {
    /* y is NaN. Return NaN, with invalid exception if y
       is a signalling NaN. */
    return y + y;
  } else if (uy == 0x3f800000)
    /* y is 1.0; return x */
    return x;
  else if ((ay & EXPBITS_SP32) > 0x4f000000) {
    /* y is infinite or so large that the result would
       overflow or underflow. Flags should be raised
       unless y is an exact infinity. */
    int yinf = (ay == EXPBITS_SP32);
    if (ypos) {
      /* y is +ve */
      if (ax == 0)
        /* abs(x) = 0.0. */
        return 0.0F;
      else if (ax < 0x3f800000) {
        /* abs(x) < 1.0 */
        if (yinf)
          return 0.0F;
        else
          return retval_errno_erange_underflow(x, y, 1);
      } else if (ax == 0x3f800000) {
        /* abs(x) = 1.0. */
        return 1.0F;
      } else {
        /* abs(x) > 1.0 */
        if (yinf)
          return infinityf_with_flags(0);
        else
          return retval_errno_erange_overflow(x, y, 1);
      }
    } else {
      /* y is -ve */
      if (ax == 0) {
        /* abs(x) = 0.0. Return +infinity. */
        return retval_errno_edom(x, y, 1);
      } else if (ax < 0x3f800000) {
        /* abs(x) < 1.0; return +infinity. */
        if (yinf)
          return infinityf_with_flags(0);
        else
          return retval_errno_erange_overflow(x, y, 1);
      } else if (ax == 0x3f800000) {
        /* abs(x) = 1.0. */
        return 1.0F;
      } else {
        /* abs(x) > 1.0 */
        if (yinf)
          return 0.0F;
        else
          return retval_errno_erange_underflow(x, y, 1);
      }
    }
  }

  /* See whether y is an integer.
     inty = 0 means not an integer.
     inty = 1 means odd integer.
     inty = 2 means even integer.
  */
  yexp = ((uy & EXPBITS_SP32) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32 + 1;
  if (yexp < 1)
    inty = 0;
  else if (yexp > 24)
    inty = 2;
  else /* 1 <= yexp <= 24 */
  {
    /* Mask out the bits of r that we don't want */
    mask = (1 << (24 - yexp)) - 1;
    if ((uy & mask) != 0)
      inty = 0;
    else if (((uy & ~mask) >> (24 - yexp)) & 0x00000001)
      inty = 1;
    else
      inty = 2;
  }

  if ((ax & EXPBITS_SP32) == EXPBITS_SP32) {
    /* x is infinity (NaN was already ruled out). */
    if (xpos) {
      /* x is +infinity */
      if (ypos)
        /* y > 0.0 */
        return x;
      else
        return 0.0F;
    } else {
      /* x is -infinity */
      if (inty == 1) {
        /* y is an odd integer */
        if (ypos)
          /* Result is -infinity */
          return x;
        else
          return -0.0F;
      } else {
        if (ypos)
          /* Result is +infinity */
          return -x;
        else
          return 0.0F;
      }
    }
  } else if (ax == 0) {
    /* x is zero */
    if (xpos) {
      /* x is +0.0 */
      if (ypos)
        /* y is positive; return +0.0 for all cases */
        return x;
      else
        /* y is negative; return +infinity with div-by-zero
           for all cases */
        return retval_errno_edom(x, y, 1);
    } else {
      /* x is -0.0 */
      if (ypos) {
        /* y is positive */
        if (inty == 1)
          /* -0.0 raised to a positive odd integer returns -0.0 */
          return x;
        else
          /* Return +0.0 */
          return -x;
      } else {
        /* y is negative */
        if (inty == 1) {
          /* -0.0 raised to a negative odd integer returns -infinity
             with div-by-zero */
          return retval_errno_edom(x, y, 2);
        } else {
          /* Return +infinity with div-by-zero */
          return retval_errno_edom(x, y, 1);
        }
      }
    }
  }

  negateres = 0;
  if (!xpos) {
    /* x is negative */
    if (inty) {
      /* It's OK because y is an integer. */
      ux = ax;
      PUT_BITS_SP32(ux, x); /* x = abs(x) */
      /* If y is odd, the result will be negative */
      negateres = (inty == 1);
    } else
      /* y is not an integer. Return a NaN. */
      return retval_errno_edom(x, y, 3);
  }

  if (ay < 0x2e800000) /* abs(y) < 2^(-34) */
  {
    /* y is close enough to zero for the result to be 1.0
       no matter what the size of x */
    return 1.0F + y;
  }

  /* Simply use double precision for computation of log2(x),
     y*log2(x) and exp2(y*log2(x)) */
  dx = x;
  dy = y;
  dlog2 = FN_PROTOTYPE(mth_i_dlog2)(dx);
  dw = dy * dlog2;
  dr = FN_PROTOTYPE(mth_i_dexp2)(dw);

  /* If dr overflowed or underflowed we need to deal with errno */
  if (dr > large) {
    /* Double dr has overflowed range of float. */
    if (negateres)
      return retval_errno_erange_overflow(x, y, -1);
    else
      return retval_errno_erange_overflow(x, y, 1);
  } else if (dr < tiny) {
    /* Double dr has underflowed range of float. */
    if (negateres)
      return retval_errno_erange_underflow(x, y, -1);
    else
      return retval_errno_erange_underflow(x, y, 1);
  } else {
    if (negateres)
      return (float)-dr;
    else
      return (float)dr;
  }
}
