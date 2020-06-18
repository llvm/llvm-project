/*
 * Single-precision vector e^x function.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "mathlib.h"
#include "v_math.h"
#if V_SUPPORTED

static const float Poly[] = {
  /*  maxerr: 0.36565 +0.5 ulp.  */
  0x1.6a6000p-10f,
  0x1.12718ep-7f,
  0x1.555af0p-5f,
  0x1.555430p-3f,
  0x1.fffff4p-2f,
};
#define C0 v_f32 (Poly[0])
#define C1 v_f32 (Poly[1])
#define C2 v_f32 (Poly[2])
#define C3 v_f32 (Poly[3])
#define C4 v_f32 (Poly[4])

#define Shift v_f32 (0x1.8p23f)
#define InvLn2 v_f32 (0x1.715476p+0f)
#define Ln2hi v_f32 (0x1.62e4p-1f)
#define Ln2lo v_f32 (0x1.7f7d1cp-20f)

VPCS_ATTR
static v_f32_t
specialcase (v_f32_t poly, v_f32_t n, v_u32_t e, v_f32_t absn)
{
  /* 2^n may overflow, break it up into s1*s2.  */
  v_u32_t b = v_cond_u32 (n <= v_f32 (0.0f)) & v_u32 (0x83000000);
  v_f32_t s1 = v_as_f32_u32 (v_u32 (0x7f000000) + b);
  v_f32_t s2 = v_as_f32_u32 (e - b);
  v_u32_t cmp = v_cond_u32 (absn > v_f32 (192.0f));
  v_f32_t r1 = s1 * s1;
  v_f32_t r0 = poly * s1 * s2;
  return v_as_f32_u32 ((cmp & v_as_u32_f32 (r1)) | (~cmp & v_as_u32_f32 (r0)));
}

VPCS_ATTR
v_f32_t
V_NAME(expf_1u) (v_f32_t x)
{
  v_f32_t n, r, scale, poly, absn, z;
  v_u32_t cmp, e;

  /* exp(x) = 2^n * poly(r), with poly(r) in [1/sqrt(2),sqrt(2)]
     x = ln2*n + r, with r in [-ln2/2, ln2/2].  */
#if 1
  z = v_fma_f32 (x, InvLn2, Shift);
  n = z - Shift;
  r = v_fma_f32 (n, -Ln2hi, x);
  r = v_fma_f32 (n, -Ln2lo, r);
  e = v_as_u32_f32 (z) << 23;
#else
  z = x * InvLn2;
  n = v_round_f32 (z);
  r = v_fma_f32 (n, -Ln2hi, x);
  r = v_fma_f32 (n, -Ln2lo, r);
  e = v_as_u32_s32 (v_round_s32 (z)) << 23;
#endif
  scale = v_as_f32_u32 (e + v_u32 (0x3f800000));
  absn = v_abs_f32 (n);
  cmp = v_cond_u32 (absn > v_f32 (126.0f));
  poly = v_fma_f32 (C0, r, C1);
  poly = v_fma_f32 (poly, r, C2);
  poly = v_fma_f32 (poly, r, C3);
  poly = v_fma_f32 (poly, r, C4);
  poly = v_fma_f32 (poly, r, v_f32 (1.0f));
  poly = v_fma_f32 (poly, r, v_f32 (1.0f));
  if (unlikely (v_any_u32 (cmp)))
    return specialcase (poly, n, e, absn);
  return scale * poly;
}
#endif
