/*
 * Double-precision vector e^x function.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "mathlib.h"
#include "v_math.h"
#if V_SUPPORTED
#include "v_exp.h"

#if V_EXP_TABLE_BITS == 7
/* maxerr: 1.88 +0.5 ulp
   rel error: 1.4337*2^-53
   abs error: 1.4299*2^-53 in [ -ln2/256, ln2/256 ].  */
#define C1 v_f64 (0x1.ffffffffffd43p-2)
#define C2 v_f64 (0x1.55555c75adbb2p-3)
#define C3 v_f64 (0x1.55555da646206p-5)
#define InvLn2 v_f64 (0x1.71547652b82fep7) /* N/ln2.  */
#define Ln2hi v_f64 (0x1.62e42fefa39efp-8) /* ln2/N.  */
#define Ln2lo v_f64 (0x1.abc9e3b39803f3p-63)
#elif V_EXP_TABLE_BITS == 8
/* maxerr: 0.54 +0.5 ulp
   rel error: 1.4318*2^-58
   abs error: 1.4299*2^-58 in [ -ln2/512, ln2/512 ].  */
#define C1 v_f64 (0x1.fffffffffffd4p-2)
#define C2 v_f64 (0x1.5555571d6b68cp-3)
#define C3 v_f64 (0x1.5555576a59599p-5)
#define InvLn2 v_f64 (0x1.71547652b82fep8)
#define Ln2hi v_f64 (0x1.62e42fefa39efp-9)
#define Ln2lo v_f64 (0x1.abc9e3b39803f3p-64)
#endif

#define N (1 << V_EXP_TABLE_BITS)
#define Tab __v_exp_data
#define IndexMask v_u64 (N - 1)
#define Shift v_f64 (0x1.8p+52)
#define Thres v_f64 (704.0)

VPCS_ATTR
static v_f64_t
specialcase (v_f64_t s, v_f64_t y, v_f64_t n)
{
  v_f64_t absn = v_abs_f64 (n);

  /* 2^(n/N) may overflow, break it up into s1*s2.  */
  v_u64_t b = v_cond_u64 (n <= v_f64 (0.0)) & v_u64 (0x6000000000000000);
  v_f64_t s1 = v_as_f64_u64 (v_u64 (0x7000000000000000) - b);
  v_f64_t s2 = v_as_f64_u64 (v_as_u64_f64 (s) - v_u64 (0x3010000000000000) + b);
  v_u64_t cmp = v_cond_u64 (absn > v_f64 (1280.0 * N));
  v_f64_t r1 = s1 * s1;
  v_f64_t r0 = v_fma_f64 (y, s2, s2) * s1;
  return v_as_f64_u64 ((cmp & v_as_u64_f64 (r1)) | (~cmp & v_as_u64_f64 (r0)));
}

VPCS_ATTR
v_f64_t
V_NAME(exp) (v_f64_t x)
{
  v_f64_t n, r, r2, s, y, z;
  v_u64_t cmp, u, e, i;

  cmp = v_cond_u64 (v_abs_f64 (x) > Thres);

  /* n = round(x/(ln2/N)).  */
  z = v_fma_f64 (x, InvLn2, Shift);
  u = v_as_u64_f64 (z);
  n = z - Shift;

  /* r = x - n*ln2/N.  */
  r = x;
  r = v_fma_f64 (-Ln2hi, n, r);
  r = v_fma_f64 (-Ln2lo, n, r);

  e = u << (52 - V_EXP_TABLE_BITS);
  i = u & IndexMask;

  /* y = exp(r) - 1 ~= r + C1 r^2 + C2 r^3 + C3 r^4.  */
  r2 = r * r;
  y = v_fma_f64 (C2, r, C1);
  y = v_fma_f64 (C3, r2, y);
  y = v_fma_f64 (y, r2, r);

  /* s = 2^(n/N).  */
  u = v_lookup_u64 (Tab, i);
  s = v_as_f64_u64 (u + e);

  if (unlikely (v_any_u64 (cmp)))
    return specialcase (s, y, n);
  return v_fma_f64 (y, s, s);
}
VPCS_ALIAS
#endif
