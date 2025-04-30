
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <math.h>
#include <common_tanf.h>

//static vmask i2opi_vec[] = {
//    vcast_vm_i_i(0, i2opi_f[0]),
//    vcast_vm_i_i(0, i2opi_f[1]),
//    vcast_vm_i_i(0, i2opi_f[2]),
//    vcast_vm_i_i(0, i2opi_f[3]),
//    vcast_vm_i_i(0, i2opi_f[4]),
//    vcast_vm_i_i(0, i2opi_f[5]),
//};

vfloat static INLINE
__reduction_slowpath(vfloat const a, vmask *h)
{
    vint2 ia, e, idx, q, p, s;
    vint2 ia_a, ia_b, p_a, p_b, hi_a, hi_b;
    vint2 hi, lo, ll, prev, prev2;

    vmask i2opi_vec[] = {
        vcast_vm_i_i(0, i2opi_f[0]),
        vcast_vm_i_i(0, i2opi_f[1]),
        vcast_vm_i_i(0, i2opi_f[2]),
        vcast_vm_i_i(0, i2opi_f[3]),
        vcast_vm_i_i(0, i2opi_f[4]),
        vcast_vm_i_i(0, i2opi_f[5]),
    };

    ia = (vint2)a;
    s = vand_vi2_vi2_vi2(ia, vcast_vi2_i(0x80000000));
    /* e = ((ia >> 23) & 0xff) - 127; */
    e = vsrl_vi2_vi2_i(ia, 23);
    e = vand_vi2_vi2_vi2(e, vcast_vi2_i(0xff));
    e = vsub_vi2_vi2_vi2(e, vcast_vi2_i(127));
    /* ia = (ia << 8) | 0x80000000; */
    ia = vsll_vi2_vi2_i(ia, 8);
    ia = vor_vi2_vi2_vi2(ia, vcast_vi2_i(0x80000000));

    /* compute x * 1/pi */
    /* idx = 6 - ((e >> 5) & 3); */
    idx = vsrl_vi2_vi2_i(e, 5);
    idx = vand_vi2_vi2_vi2(idx, vcast_vi2_i(3));
    idx = vsub_vi2_vi2_vi2(vcast_vi2_i(6), idx);

    ia_a = vsrl64_vi2_vi2_i(ia, 32);
    ia_b = ia;
    hi_a = vcast_vi2_i(0);
    hi_b = vcast_vi2_i(0);

    q = vcast_vi2_i(0);
    for (int i = 0; i < 6; i++) {
        p_a = vmulu_vi2_vi2_vi2((vint2)i2opi_vec[i], ia_a);
        p_b = vmulu_vi2_vi2_vi2((vint2)i2opi_vec[i], ia_b);
        p_a = vadd64_vi2_vi2_vi2(p_a, hi_a);
        p_b = vadd64_vi2_vi2_vi2(p_b, hi_b);

        hi_a = vsrl64_vi2_vi2_i(p_a, 32);
        hi_b = vsrl64_vi2_vi2_i(p_b, 32);

        p_a = vsll64_vi2_vi2_i(p_a, 32);
        p_b = vand_vi2_vi2_vi2(p_b, vcast_vm_i_i(0, 0xffffffff));

        p = vor_vi2_vi2_vi2(p_a, p_b);

        vopmask m = veq_vo_vi2_vi2(idx, q);
        hi = vsel_vi2_vo_vi2_vi2(m, p, hi);
        lo = vsel_vi2_vo_vi2_vi2(m, prev, lo);
        ll = vsel_vi2_vo_vi2_vi2(m, prev2, ll);

        prev2 = prev;
        prev = p;

        q = vadd_vi2_vi2_vi2(q, vcast_vi2_i(1));
    }
    p = vor_vi2_vi2_vi2(vsll64_vi2_vi2_i(hi_a, 32), hi_b);

    vopmask m = veq_vo_vi2_vi2(idx, q);
    hi = vsel_vi2_vo_vi2_vi2(m, p, hi);
    lo = vsel_vi2_vo_vi2_vi2(m, prev, lo);
    ll = vsel_vi2_vo_vi2_vi2(m, prev2, ll);

    e = vand_vi2_vi2_vi2(e, vcast_vi2_i(31));

    union {
        vint2 v;
        uint32_t t[sizeof(vint2) / sizeof(uint32_t)];
    } ue, uhi, ulo, ull, uh, ur;
    ue.v = e; uhi.v = hi; ulo.v = lo; ull.v = ll;
    for (unsigned i = 0; i < sizeof(vint2) / sizeof(uint32_t); i++) {
        uint32_t e = ue.t[i], q;
        uint64_t p;
        p = (uint64_t)uhi.t[i] << 32;
        p |= ulo.t[i];

        if (e) {
            q = 32 - e;
            p = (p << e) | (ull.t[i] >> q);
        }

        q = (uhi.t[i] << e) & 0x80000000;
        p &= 0x7fffffffffffffffULL;

        if (p & 0x4000000000000000ULL) {
            p |= 0x8000000000000000ULL;
            q ^= 0x80000000;
        }
        uh.t[i] = q;

        double d = (double)(int64_t)p;
        d *= PI_2_M64;
        float r = (float)d;
        ur.t[i] = float_as_int(r);
    }
    vstore_v_p_vf((float*)h, (vfloat)uh.v);
    return (vfloat)vxor_vi2_vi2_vi2(ur.v, s);
}

vfloat static INLINE
__tan_kernel(vfloat const a, vint2 const h)
{
    vfloat s, r, rd, t;
    vopmask cmp;

    s = vmul_vf_vf_vf(a, a);
    r = vcast_vf_f(A_F);
    r = vfma_vf_vf_vf_vf(r, s, vcast_vf_f(B_F));
    r = vfma_vf_vf_vf_vf(r, s, vcast_vf_f(C_F));
    r = vfma_vf_vf_vf_vf(r, s, vcast_vf_f(D_F));
    r = vfma_vf_vf_vf_vf(r, s, vcast_vf_f(E_F));
    r = vfma_vf_vf_vf_vf(r, s, vcast_vf_f(F_F));
    t = vmul_vf_vf_vf(s, a);
    r = vfma_vf_vf_vf_vf(r, t, a);

    rd = vdiv_vf_vf_vf(vcast_vf_f(-1.0f), r);
    cmp = veq_vo_vi2_vi2((vint2)h, vcast_vi2_i(0));
    r = vsel_vf_vo_vf_vf(cmp, r, rd);

    return r;
}

vfloat static INLINE
__tan_f_vec(vfloat const x)
{

    vfloat a, k, r;
    vopmask m;
    vint2 p, h;

    k = vfma_vf_vf_vf_vf(x, vcast_vf_f(_2_OVER_PI_F), vcast_vf_f(12582912.0f));
    h = vsll_vi2_vi2_i((vint2)k, 31);
    k = vsub_vf_vf_vf(k, vcast_vf_f(12582912.0f));

    a = vfma_vf_vf_vf_vf(k, vcast_vf_f(-PI_2_HI_F), x);
    a = vfma_vf_vf_vf_vf(k, vcast_vf_f(-PI_2_MI_F), a);
    a = vfma_vf_vf_vf_vf(k, vcast_vf_f(-PI_2_LO_F), a);

    r = __tan_kernel(a, h);

    p = vand_vi2_vi2_vi2((vint2)x, vcast_vi2_i(0x7fffffff));
    m = vgt_vo_vi2_vi2(p, (vint2)vcast_vf_f(THRESHOLD_F));
    if (__builtin_expect(!vtestz_i_vo(m), 0)) {
        vfloat res;
        vopmask ninf;
        vmask half;

        res = __reduction_slowpath(x, &half);
        res = __tan_kernel(res, half);
        ninf = vgt_vo_vi2_vi2(vcast_vi2_i(0x7f800000), p);
        res = vsel_vf_vo_vf_vf(ninf, res, vmul_vf_vf_vf(x, vcast_vf_f(0.0f)));

        r = vsel_vf_vo_vf_vf(m, res, r);
    }

    return r;
}
