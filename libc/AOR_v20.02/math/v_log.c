/*
 * Double-precision vector log(x) function.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "mathlib.h"
#include "v_math.h"
#include "v_log.h"
#if V_SUPPORTED

/* Worst-case error: 1.17 + 0.5 ulp.  */

static const f64_t Poly[] = {
  /* rel error: 0x1.6272e588p-56 in [ -0x1.fc1p-9 0x1.009p-8 ].  */
  -0x1.ffffffffffff7p-2,
   0x1.55555555170d4p-2,
  -0x1.0000000399c27p-2,
   0x1.999b2e90e94cap-3,
  -0x1.554e550bd501ep-3,
};

#define A0 v_f64 (Poly[0])
#define A1 v_f64 (Poly[1])
#define A2 v_f64 (Poly[2])
#define A3 v_f64 (Poly[3])
#define A4 v_f64 (Poly[4])
#define Ln2 v_f64 (0x1.62e42fefa39efp-1)
#define N (1 << V_LOG_TABLE_BITS)
#define OFF v_u64 (0x3fe6900900000000)

struct entry
{
  v_f64_t invc;
  v_f64_t logc;
};

static inline struct entry
lookup (v_u64_t i)
{
  struct entry e;
#ifdef SCALAR
  e.invc = __v_log_data[i].invc;
  e.logc = __v_log_data[i].logc;
#else
  e.invc[0] = __v_log_data[i[0]].invc;
  e.logc[0] = __v_log_data[i[0]].logc;
  e.invc[1] = __v_log_data[i[1]].invc;
  e.logc[1] = __v_log_data[i[1]].logc;
#endif
  return e;
}

VPCS_ATTR
__attribute__ ((noinline)) static v_f64_t
specialcase (v_f64_t x, v_f64_t y, v_u64_t cmp)
{
  return v_call_f64 (log, x, y, cmp);
}

VPCS_ATTR
v_f64_t
V_NAME(log) (v_f64_t x)
{
  v_f64_t z, r, r2, p, y, kd, hi;
  v_u64_t ix, iz, tmp, top, i, cmp;
  v_s64_t k;
  struct entry e;

  ix = v_as_u64_f64 (x);
  top = ix >> 48;
  cmp = v_cond_u64 (top - v_u64 (0x0010) >= v_u64 (0x7ff0 - 0x0010));

  /* x = 2^k z; where z is in range [OFF,2*OFF) and exact.
     The range is split into N subintervals.
     The ith subinterval contains z and c is near its center.  */
  tmp = ix - OFF;
  i = (tmp >> (52 - V_LOG_TABLE_BITS)) % N;
  k = v_as_s64_u64 (tmp) >> 52; /* arithmetic shift */
  iz = ix - (tmp & v_u64 (0xfffULL << 52));
  z = v_as_f64_u64 (iz);
  e = lookup (i);

  /* log(x) = log1p(z/c-1) + log(c) + k*Ln2.  */
  r = v_fma_f64 (z, e.invc, v_f64 (-1.0));
  kd = v_to_f64_s64 (k);

  /* hi = r + log(c) + k*Ln2.  */
  hi = v_fma_f64 (kd, Ln2, e.logc + r);
  /* y = r2*(A0 + r*A1 + r2*(A2 + r*A3 + r2*A4)) + hi.  */
  r2 = r * r;
  y = v_fma_f64 (A3, r, A2);
  p = v_fma_f64 (A1, r, A0);
  y = v_fma_f64 (A4, r2, y);
  y = v_fma_f64 (y, r2, p);
  y = v_fma_f64 (y, r2, hi);

  if (unlikely (v_any_u64 (cmp)))
    return specialcase (x, y, cmp);
  return y;
}
VPCS_ALIAS
#endif
