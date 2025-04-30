
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <common.h>
/* #include "../scalar/log_scalar.h" */

vfloat __attribute__ ((noinline))
log_vec(vfloat const a_input)
{
    vfloat m, m2, e, t;
    vint2 u;
    vopmask o;

    m = (vfloat)vsub_vi2_vi2_vi2((vint2)a_input, *(vint2*)MAGIC_F);
    e = (vfloat)vsra_vi2_vi2_i((vint2)m, 23);
    m = (vfloat)vand_vi2_vi2_vi2((vint2)m, *(vint2*)MANTISSA_MASK);
    m = (vfloat)vadd_vi2_vi2_vi2((vint2)m, *(vint2*)MAGIC_F);

    e = vcast_vf_vi2((vint2)e);

    m = vsub_vf_vf_vf(m, *(vfloat*)ONE_F);
    m2 = vmul_vf_vf_vf(m, m);

    t = *(vfloat*)c0;
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c1);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c2);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c3);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c4);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c5);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c6);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c7);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c8);
    t = vfma_vf_vf_vf_vf(t, m, *(vfloat*)c9);
    t = vfma_vf_vf_vf_vf(t, m2, m);
    t = vfma_vf_vf_vf_vf(e, *(vfloat*)LOG_2_F, t);

    // slowpath
    u = vadd_vi2_vi2_vi2((vint2)a_input, vcast_vi2_i(0x800000));
    o = vgt_vo_vi2_vi2(vcast_vi2_i(0x1000000), u);
    if (__builtin_expect(!vtestz_i_vo(o), 0)) {
        vopmask inf_mask = veq_vo_vi2_vi2((vint2)a_input, vcast_vi2_i(0x7f800000));
        vopmask den_mask = vgt_vo_vi2_vi2(vcast_vi2_i(0x800000), (vint2)a_input);
        vopmask neg_mask = vgt_vo_vi2_vi2(vcast_vi2_i(0), (vint2)a_input);
        vopmask zer_mask = veq_vo_vf_vf(vcast_vf_f(0.0f), a_input);
        vopmask nan_mask = vneq_vo_vf_vf(a_input, a_input);

        vfloat inf_out = vcast_vf_f(PINF);
        vfloat neg_out = vcast_vf_f(CANONICAL_NAN);
        vfloat zer_out = vcast_vf_f(NINF);
        vfloat nan_out = vadd_vf_vf_vf(a_input, a_input);

        vfloat a2p24 = vmul_vf_vf_vf(a_input, vcast_vf_f(TWO_TO_24_F));

        m = (vfloat)vsub_vi2_vi2_vi2((vint2)a2p24, *(vint2*)MAGIC_F);
        e = (vfloat)vsub_vi2_vi2_vi2(vsra_vi2_vi2_i((vint2)m, 23), vcast_vi2_i(24));
        m = (vfloat)vand_vi2_vi2_vi2((vint2)m, *(vint2*)MANTISSA_MASK);
        m = (vfloat)vadd_vi2_vi2_vi2((vint2)m, *(vint2*)MAGIC_F);

        e = vcast_vf_vi2((vint2)e);

        m = vsub_vf_vf_vf(m, *(vfloat*)ONE_F);
        m2 = vmul_vf_vf_vf(m, m);

        vfloat den_out = *(vfloat*)c0;
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c1);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c2);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c3);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c4);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c5);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c6);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c7);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c8);
        den_out = vfma_vf_vf_vf_vf(den_out, m, *(vfloat*)c9);
        den_out = vfma_vf_vf_vf_vf(den_out, m2, m);
        den_out = vfma_vf_vf_vf_vf(e, *(vfloat*)LOG_2_F, den_out);

        t = vsel_vf_vo_vf_vf(inf_mask, inf_out, t);
        t = vsel_vf_vo_vf_vf(den_mask, den_out, t);
        t = vsel_vf_vo_vf_vf(neg_mask, neg_out, t);
        t = vsel_vf_vo_vf_vf(zer_mask, zer_out, t);
        t = vsel_vf_vo_vf_vf(nan_mask, nan_out, t);
        return t;
    }

    return t;
}

