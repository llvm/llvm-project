/*
 * Single-precision vector powf function.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "mathlib.h"
#include "v_math.h"
#if V_SUPPORTED

#define Min v_u32 (0x00800000)
#define Max v_u32 (0x7f800000)
#define SBITS 5
#define Tlog v__powf_log2_data.tab
#define Texp v__exp2f_data.tab
#define A v__powf_log2_data.poly
#define C v__exp2f_data.poly
#define LOGDEG 4

#if LOGDEG == 5
/* 1.01 ulp */
#define OFF v_u32 (0x3f330000)
#define TBITS 4
#elif LOGDEG == 4
/* 2.6 ulp ~ 0.5 + 2^24 (128*Ln2*relerr_log2 + relerr_exp2) */
#define OFF v_u32 (0x3f35d000)
#define TBITS 5
#endif

#define V_EXP2F_TABLE_BITS SBITS
#define V_EXP2F_POLY_ORDER 3
struct v_exp2f_data
{
  uint64_t tab[1 << V_EXP2F_TABLE_BITS];
  double poly[V_EXP2F_POLY_ORDER];
};

#define V_POWF_LOG2_TABLE_BITS TBITS
#define V_POWF_LOG2_POLY_ORDER LOGDEG
#define SCALE ((double) (1 << SBITS))
struct v_powf_log2_data
{
  struct
  {
    double invc, logc;
  } tab[1 << V_POWF_LOG2_TABLE_BITS];
  double poly[V_POWF_LOG2_POLY_ORDER];
};

static const struct v_powf_log2_data v__powf_log2_data = {
#if LOGDEG == 5
  .tab = {
{ 0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2 * SCALE },
{ 0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2 * SCALE },
{ 0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2 * SCALE },
{ 0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2 * SCALE },
{ 0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2 * SCALE },
{ 0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3 * SCALE },
{ 0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3 * SCALE },
{ 0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4 * SCALE },
{ 0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5 * SCALE },
{ 0x1p+0, 0x0p+0 * SCALE },
{ 0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4 * SCALE },
{ 0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3 * SCALE },
{ 0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3 * SCALE },
{ 0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2 * SCALE },
{ 0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2 * SCALE },
{ 0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2 * SCALE },
  },
/* rel err: 1.46 * 2^-32 */
  .poly = {
0x1.27616c9496e0bp-2 * SCALE, -0x1.71969a075c67ap-2 * SCALE,
0x1.ec70a6ca7baddp-2 * SCALE, -0x1.7154748bef6c8p-1 * SCALE,
0x1.71547652ab82bp0 * SCALE,
  }
#elif LOGDEG == 4
  .tab = {
{0x1.6489890582816p+0, -0x1.e960f97b22702p-2 * SCALE},
{0x1.5cf19b35e3472p+0, -0x1.c993406cd4db6p-2 * SCALE},
{0x1.55aac0e956d65p+0, -0x1.aa711d9a7d0f3p-2 * SCALE},
{0x1.4eb0022977e01p+0, -0x1.8bf37bacdce9bp-2 * SCALE},
{0x1.47fcccda1dd1fp+0, -0x1.6e13b3519946ep-2 * SCALE},
{0x1.418ceabab68c1p+0, -0x1.50cb8281e4089p-2 * SCALE},
{0x1.3b5c788f1edb3p+0, -0x1.341504a237e2bp-2 * SCALE},
{0x1.3567de48e9c9ap+0, -0x1.17eaab624ffbbp-2 * SCALE},
{0x1.2fabc80fd19bap+0, -0x1.f88e708f8c853p-3 * SCALE},
{0x1.2a25200ce536bp+0, -0x1.c24b6da113914p-3 * SCALE},
{0x1.24d108e0152e3p+0, -0x1.8d02ee397cb1dp-3 * SCALE},
{0x1.1facd8ab2fbe1p+0, -0x1.58ac1223408b3p-3 * SCALE},
{0x1.1ab614a03efdfp+0, -0x1.253e6fd190e89p-3 * SCALE},
{0x1.15ea6d03af9ffp+0, -0x1.e5641882c12ffp-4 * SCALE},
{0x1.1147b994bb776p+0, -0x1.81fea712926f7p-4 * SCALE},
{0x1.0ccbf650593aap+0, -0x1.203e240de64a3p-4 * SCALE},
{0x1.0875408477302p+0, -0x1.8029b86a78281p-5 * SCALE},
{0x1.0441d42a93328p+0, -0x1.85d713190fb9p-6 * SCALE},
{0x1p+0, 0x0p+0 * SCALE},
{0x1.f1d006c855e86p-1, 0x1.4c1cc07312997p-5 * SCALE},
{0x1.e28c3341aa301p-1, 0x1.5e1848ccec948p-4 * SCALE},
{0x1.d4bdf9aa64747p-1, 0x1.04cfcb7f1196fp-3 * SCALE},
{0x1.c7b45a24e5803p-1, 0x1.582813d463c21p-3 * SCALE},
{0x1.bb5f5eb2ed60ap-1, 0x1.a936fa68760ccp-3 * SCALE},
{0x1.afb0bff8fe6b4p-1, 0x1.f81bc31d6cc4ep-3 * SCALE},
{0x1.a49badf7ab1f5p-1, 0x1.2279a09fae6b1p-2 * SCALE},
{0x1.9a14a111fc4c9p-1, 0x1.47ec0b6df5526p-2 * SCALE},
{0x1.901131f5b2fdcp-1, 0x1.6c71762280f1p-2 * SCALE},
{0x1.8687f73f6d865p-1, 0x1.90155070798dap-2 * SCALE},
{0x1.7d7067eb77986p-1, 0x1.b2e23b1d3068cp-2 * SCALE},
{0x1.74c2c1cf97b65p-1, 0x1.d4e21b0daa86ap-2 * SCALE},
{0x1.6c77f37cff2a1p-1, 0x1.f61e2a2f67f3fp-2 * SCALE},
  },
/* rel err: 1.5 * 2^-30 */
  .poly = {
 -0x1.6ff5daa3b3d7cp-2 * SCALE,
 0x1.ec81d03c01aebp-2 * SCALE,
 -0x1.71547bb43f101p-1 * SCALE,
 0x1.7154764a815cbp0 * SCALE,
  }
#endif
};

static const struct v_exp2f_data v__exp2f_data = {
  .tab = {
0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
  },
/* rel err: 1.69 * 2^-34 */
  .poly = {
0x1.c6af84b912394p-5/SCALE/SCALE/SCALE, 0x1.ebfce50fac4f3p-3/SCALE/SCALE, 0x1.62e42ff0c52d6p-1/SCALE
  },
};

VPCS_ATTR
__attribute__ ((noinline)) static v_f32_t
specialcase (v_f32_t x, v_f32_t y, v_f32_t ret, v_u32_t cmp)
{
  return v_call2_f32 (powf, x, y, ret, cmp);
}

VPCS_ATTR
v_f32_t
V_NAME(powf) (v_f32_t x, v_f32_t y)
{
  v_u32_t u, tmp, cmp, i, top, iz;
  v_s32_t k;
  v_f32_t ret;

  u = v_as_u32_f32 (x);
  cmp = v_cond_u32 (u - Min >= Max - Min);
  tmp = u - OFF;
  i = (tmp >> (23 - TBITS)) % (1 << TBITS);
  top = tmp & 0xff800000;
  iz = u - top;
  k = v_as_s32_u32 (top) >> (23 - SBITS); /* arithmetic shift */

  for (int lane = 0; lane < v_lanes32 (); lane++)
    {
      uint32_t si, siz;
      int32_t sk;
      float sy;

      /* Use double precision for each lane.  */
      double invc, logc, z, r, p, y0, logx, ylogx, kd, s;
      uint64_t ki, t;

      si = v_get_u32 (i, lane);
      siz = v_get_u32 (iz, lane);
      sk = v_get_s32 (k, lane);
      sy = v_get_f32 (y, lane);

      invc = Tlog[si].invc;
      logc = Tlog[si].logc;
      z = (double) as_f32_u32 (siz);

      /* log2(x) = log1p(z/c-1)/ln2 + log2(c) + k */
      r = __builtin_fma (z, invc, -1.0);
      y0 = logc + (double) sk;

      /* Polynomial to approximate log1p(r)/ln2.  */
#if LOGDEG == 5
      logx = A[0];
      logx = r * logx + A[1];
      logx = r * logx + A[2];
      logx = r * logx + A[3];
      logx = r * logx + A[4];
      logx = r * logx + y0;
#elif LOGDEG == 4
      logx = A[0];
      logx = r * logx + A[1];
      logx = r * logx + A[2];
      logx = r * logx + A[3];
      logx = r * logx + y0;
#endif
      ylogx = sy * logx;
      v_set_u32 (&cmp, lane,
		 (as_u64_f64 (ylogx) >> 47 & 0xffff)
		     >= as_u64_f64 (126.0 * (1 << SBITS)) >> 47
		   ? 1
		   : v_get_u32 (cmp, lane));

      /* N*x = k + r with r in [-1/2, 1/2] */
#if TOINT_INTRINSICS
      kd = roundtoint (ylogx); /* k */
      ki = converttoint (ylogx);
#else
# define SHIFT 0x1.8p52
      kd = eval_as_double (ylogx + SHIFT);
      ki = asuint64 (kd);
      kd -= SHIFT;
#endif
      r = ylogx - kd;

      /* exp2(x) = 2^(k/N) * 2^r ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
      t = Texp[ki % (1 << SBITS)];
      t += ki << (52 - SBITS);
      s = as_f64_u64 (t);
      p = C[0];
      p = __builtin_fma (p, r, C[1]);
      p = __builtin_fma (p, r, C[2]);
      p = __builtin_fma (p, s * r, s);

      v_set_f32 (&ret, lane, p);
    }
  if (unlikely (v_any_u32 (cmp)))
    return specialcase (x, y, ret, cmp);
  return ret;
}
VPCS_ALIAS
#endif
