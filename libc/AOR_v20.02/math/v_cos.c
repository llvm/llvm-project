/*
 * Double-precision vector cos function.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "mathlib.h"
#include "v_math.h"
#if V_SUPPORTED

static const double Poly[] = {
/* worst-case error is 3.5 ulp.
   abs error: 0x1.be222a58p-53 in [-pi/2, pi/2].  */
-0x1.9f4a9c8b21dc9p-41,
 0x1.60e88a10163f2p-33,
-0x1.ae6361b7254e7p-26,
 0x1.71de382e8d62bp-19,
-0x1.a01a019aeb4ffp-13,
 0x1.111111110b25ep-7,
-0x1.55555555554c3p-3,
};

#define C7 v_f64 (Poly[0])
#define C6 v_f64 (Poly[1])
#define C5 v_f64 (Poly[2])
#define C4 v_f64 (Poly[3])
#define C3 v_f64 (Poly[4])
#define C2 v_f64 (Poly[5])
#define C1 v_f64 (Poly[6])

#define InvPi v_f64 (0x1.45f306dc9c883p-2)
#define HalfPi v_f64 (0x1.921fb54442d18p+0)
#define Pi1 v_f64 (0x1.921fb54442d18p+1)
#define Pi2 v_f64 (0x1.1a62633145c06p-53)
#define Pi3 v_f64 (0x1.c1cd129024e09p-106)
#define Shift v_f64 (0x1.8p52)
#define RangeVal v_f64 (0x1p23)
#define AbsMask v_u64 (0x7fffffffffffffff)

VPCS_ATTR
__attribute__ ((noinline)) static v_f64_t
specialcase (v_f64_t x, v_f64_t y, v_u64_t cmp)
{
  return v_call_f64 (cos, x, y, cmp);
}

VPCS_ATTR
v_f64_t
V_NAME(cos) (v_f64_t x)
{
  v_f64_t n, r, r2, y;
  v_u64_t odd, cmp;

  r = v_as_f64_u64 (v_as_u64_f64 (x) & AbsMask);
  cmp = v_cond_u64 (v_as_u64_f64 (r) >= v_as_u64_f64 (RangeVal));

  /* n = rint((|x|+pi/2)/pi) - 0.5.  */
  n = v_fma_f64 (InvPi, r + HalfPi, Shift);
  odd = v_as_u64_f64 (n) << 63;
  n -= Shift;
  n -= v_f64 (0.5);

  /* r = |x| - n*pi  (range reduction into -pi/2 .. pi/2).  */
  r = v_fma_f64 (-Pi1, n, r);
  r = v_fma_f64 (-Pi2, n, r);
  r = v_fma_f64 (-Pi3, n, r);

  /* sin(r) poly approx.  */
  r2 = r * r;
  y = v_fma_f64 (C7, r2, C6);
  y = v_fma_f64 (y, r2, C5);
  y = v_fma_f64 (y, r2, C4);
  y = v_fma_f64 (y, r2, C3);
  y = v_fma_f64 (y, r2, C2);
  y = v_fma_f64 (y, r2, C1);
  y = v_fma_f64 (y * r2, r, r);

  /* sign.  */
  y = v_as_f64_u64 (v_as_u64_f64 (y) ^ odd);

  if (unlikely (v_any_u64 (cmp)))
    return specialcase (x, y, cmp);
  return y;
}
VPCS_ALIAS
#endif
