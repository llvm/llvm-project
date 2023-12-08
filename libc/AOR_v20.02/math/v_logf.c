/*
 * Single-precision vector log function.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "mathlib.h"
#include "v_math.h"
#if V_SUPPORTED

static const float Poly[] = {
  /* 3.34 ulp error */
  -0x1.3e737cp-3f, 0x1.5a9aa2p-3f, -0x1.4f9934p-3f, 0x1.961348p-3f,
  -0x1.00187cp-2f, 0x1.555d7cp-2f, -0x1.ffffc8p-2f,
};
#define P7 v_f32 (Poly[0])
#define P6 v_f32 (Poly[1])
#define P5 v_f32 (Poly[2])
#define P4 v_f32 (Poly[3])
#define P3 v_f32 (Poly[4])
#define P2 v_f32 (Poly[5])
#define P1 v_f32 (Poly[6])

#define Ln2 v_f32 (0x1.62e43p-1f) /* 0x3f317218 */
#define Min v_u32 (0x00800000)
#define Max v_u32 (0x7f800000)
#define Mask v_u32 (0x007fffff)
#define Off v_u32 (0x3f2aaaab) /* 0.666667 */

VPCS_ATTR
__attribute__ ((noinline)) static v_f32_t
specialcase (v_f32_t x, v_f32_t y, v_u32_t cmp)
{
  /* Fall back to scalar code.  */
  return v_call_f32 (logf, x, y, cmp);
}

VPCS_ATTR
v_f32_t
V_NAME(logf) (v_f32_t x)
{
  v_f32_t n, p, q, r, r2, y;
  v_u32_t u, cmp;

  u = v_as_u32_f32 (x);
  cmp = v_cond_u32 (u - Min >= Max - Min);

  /* x = 2^n * (1+r), where 2/3 < 1+r < 4/3 */
  u -= Off;
  n = v_to_f32_s32 (v_as_s32_u32 (u) >> 23); /* signextend */
  u &= Mask;
  u += Off;
  r = v_as_f32_u32 (u) - v_f32 (1.0f);

  /* y = log(1+r) + n*ln2.  */
  r2 = r * r;
  /* n*ln2 + r + r2*(P1 + r*P2 + r2*(P3 + r*P4 + r2*(P5 + r*P6 + r2*P7))).  */
  p = v_fma_f32 (P6, r, P5);
  q = v_fma_f32 (P4, r, P3);
  y = v_fma_f32 (P2, r, P1);
  p = v_fma_f32 (P7, r2, p);
  q = v_fma_f32 (p, r2, q);
  y = v_fma_f32 (q, r2, y);
  p = v_fma_f32 (Ln2, n, r);
  y = v_fma_f32 (y, r2, p);

  if (unlikely (v_any_u32 (cmp)))
    return specialcase (x, y, cmp);
  return y;
}
VPCS_ALIAS
#endif
