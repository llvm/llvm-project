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

#define USE_VALF_WITH_FLAGS
#define USE_NAN_WITH_FLAGS
#define USE_HANDLE_ERRORF
#include "libm_inlines_amd.h"
#undef USE_VALF_WITH_FLAGS
#undef USE_NAN_WITH_FLAGS
#undef USE_HANDLE_ERRORF

#include "libm_errno_amd.h"

float FN_PROTOTYPE(mth_i_atan)(float fx)
{

  /* Some constants and split constants. */

  static double piby2 = 1.5707963267948966e+00; /* 0x3ff921fb54442d18 */

  double c, v, s, q, z;
  unsigned int xnan;

  double x = fx;

  /* Find properties of argument fx. */

  __UINT8_T ux, aux, xneg;

  GET_BITS_DP64(x, ux);
  aux = ux & ~SIGNBIT_DP64;
  xneg = ux & SIGNBIT_DP64;

  v = x;
  if (xneg)
    v = -x;

  /* Argument reduction to range [-7/16,7/16] */

  if (aux < 0x3fdc000000000000) /* v < 7./16. */
  {
    x = v;
    c = 0.0;
  } else if (aux < 0x3fe6000000000000) /* v < 11./16. */
  {
    x = (2.0 * v - 1.0) / (2.0 + v);
    /* c = arctan(0.5) */
    c = 4.63647609000806093515e-01;    /* 0x3fddac670561bb4f */
  } else if (aux < 0x3ff3000000000000) /* v < 19./16. */
  {
    x = (v - 1.0) / (1.0 + v);
    /* c = arctan(1.) */
    c = 7.85398163397448278999e-01;    /* 0x3fe921fb54442d18 */
  } else if (aux < 0x4003800000000000) /* v < 39./16. */
  {
    x = (v - 1.5) / (1.0 + 1.5 * v);
    /* c = arctan(1.5) */
    c = 9.82793723247329054082e-01; /* 0x3fef730bd281f69b */
  } else {

    xnan = (aux > PINFBITPATT_DP64);

    if (xnan) {
      /* x is NaN */
      return x + x; /* Raise invalid if it's a signalling NaN */
    } else if (v > 0x4c80000000000000) { /* abs(x) > 2^26 => arctan(1/x) is
                                            insignificant compared to piby2 */
      if (xneg)
        return valf_with_flags((float)-piby2, AMD_F_INEXACT);
      else
        return valf_with_flags((float)piby2, AMD_F_INEXACT);
    }

    x = -1.0 / v;
    /* c = arctan(infinity) */
    c = 1.57079632679489655800e+00; /* 0x3ff921fb54442d18 */
  }

  /* Core approximation: Remez(2,2) on [-7/16,7/16] */

  s = x * x;
  q = x * s * (0.296528598819239217902158651186e0 +
               (0.192324546402108583211697690500e0 +
                0.470677934286149214138357545549e-2 * s) *
                   s) /
      (0.889585796862432286486651434570e0 +
       (0.111072499995399550138837673349e1 +
        0.299309699959659728404442796915e0 * s) *
           s);

  z = c - (q - x);

  if (xneg)
    z = -z;
  return (float)z;
}
