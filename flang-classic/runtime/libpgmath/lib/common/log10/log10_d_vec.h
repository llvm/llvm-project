
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <common.h>

static inline __attribute__ ((always_inline))
vdouble __log10_d_kernel(vdouble m, vdouble e)
{
    e = vmul_vd_vd_vd(e, *(vdouble*)LOG10_2);
    m = vsub_vd_vd_vd(m, *(vdouble*)ONE_D);

    vdouble m2 = vmul_vd_vd_vd(m, m);
    vdouble m4 = vmul_vd_vd_vd(m2, m2);
    vdouble m5 = vmul_vd_vd_vd(m4, m);
    vdouble m8 = vmul_vd_vd_vd(m4, m4);

    vdouble t0 = vfma_vd_vd_vd_vd(*(vdouble*)c0, m, *(vdouble*)c1);
    vdouble t1 = vfma_vd_vd_vd_vd(*(vdouble*)c2, m, *(vdouble*)c3);
    vdouble t2 = vfma_vd_vd_vd_vd(*(vdouble*)c4, m, *(vdouble*)c5);
    vdouble t3 = vfma_vd_vd_vd_vd(*(vdouble*)c6, m, *(vdouble*)c7);
    vdouble t4 = vfma_vd_vd_vd_vd(*(vdouble*)c8, m, *(vdouble*)c9);
    vdouble t5 = vfma_vd_vd_vd_vd(*(vdouble*)c10, m, *(vdouble*)c11);
    vdouble t6 = vfma_vd_vd_vd_vd(*(vdouble*)c12, m, *(vdouble*)c13);
    vdouble t7 = vfma_vd_vd_vd_vd(*(vdouble*)c14, m, *(vdouble*)c15);

    vdouble t = *(vdouble*)c16;
    t = vfma_vd_vd_vd_vd(t, m, *(vdouble*)c17);
    t = vfma_vd_vd_vd_vd(t, m, *(vdouble*)c18);
    t = vfma_vd_vd_vd_vd(t, m, *(vdouble*)c19);
    t = vfma_vd_vd_vd_vd(t, m, e);

    t0 = vfma_vd_vd_vd_vd(t0, m2, t1);
    t2 = vfma_vd_vd_vd_vd(t2, m2, t3);
    t4 = vfma_vd_vd_vd_vd(t4, m2, t5);
    t6 = vfma_vd_vd_vd_vd(t6, m2, t7);
    t0 = vfma_vd_vd_vd_vd(t0, m4, t2);
    t4 = vfma_vd_vd_vd_vd(t4, m4, t6);
    t0 = vfma_vd_vd_vd_vd(t0, m8, t4);

    t = vfma_vd_vd_vd_vd(t0, m5, t);

    return t;
}

vdouble __attribute__ ((noinline))
log10_d_vec(vdouble const a_input)
{
    vdouble m, e, t;

#ifdef __AVX512VL__
    vdouble b;
    m = vgetmant_vd_vd(a_input);
    e = vgetexp_vd_vd(a_input);
    b = vgetexp_vd_vd(m);
    e = vsub_vd_vd_vd(e, b);
#else
    vint ei;
    m = (vdouble)vsub64_vi2_vi2_vi2((vint2)a_input, *(vint2*)THRESHOLD);
    ei = vhi64_vi_vi2((vint2)m);
    ei = vsra_vi_vi_i(ei, 20);
    m = (vdouble)vand_vi2_vi2_vi2((vint2)m, *(vint2*)MANTISSA_MASK);
    m = (vdouble)vadd64_vi2_vi2_vi2((vint2)m, *(vint2*)THRESHOLD);
    e = vcast_vd_vi(ei);
#endif

    t = __log10_d_kernel(m, e);

#ifndef __AVX512VL__
    // slowpath
    const vint2 den1 = vcast_vm_i_i(0x100000, 0);
    const vint2 den2 = vsll64_vi2_vi2_i(den1, 1);
    const vint2 u = vadd64_vi2_vi2_vi2((vint2)a_input, den1);
    const vopmask o = vgt64_vo_vi2_vi2(den2, u);
    if (__builtin_expect(!vtestz_i_vo(o), 0)) {
        vopmask inf_mask = veq64_vo_vi2_vi2((vint2)a_input, *(vint2*)PINF);
        vopmask den_mask = vgt64_vo_vi2_vi2(den1, (vint2)a_input);
        vopmask neg_mask = vgt64_vo_vi2_vi2(vcast_vi2_i(0), (vint2)a_input);
        vopmask zer_mask = veq_vo_vd_vd((vdouble)vcast_vi2_i(0), a_input);
        vopmask nan_mask = vneq_vo_vd_vd(a_input, a_input);

        vdouble inf_out = *(vdouble*)PINF;
        vdouble neg_out = *(vdouble*)CANONICAL_NAN;
        vdouble zer_out = *(vdouble*)NINF;
        vdouble nan_out = vadd_vd_vd_vd(a_input, a_input);

        vdouble a2p53 = vmul_vd_vd_vd(a_input, vcast_vd_d(TWO_TO_53));
        m = (vdouble)vsub64_vi2_vi2_vi2((vint2)a2p53, *(vint2*)THRESHOLD);
        ei = vhi64_vi_vi2((vint2)m);
        ei = vsub_vi_vi_vi(vsra_vi_vi_i(ei, 20), vcast_vi_i(53));
        m = (vdouble)vand_vi2_vi2_vi2((vint2)m, *(vint2*)MANTISSA_MASK);
        m = (vdouble)vadd64_vi2_vi2_vi2((vint2)m, *(vint2*)THRESHOLD);
        e = vcast_vd_vi(ei);

        vdouble den_out = __log10_d_kernel(m, e);

        t = vsel_vd_vo_vd_vd(inf_mask, inf_out, t);
        t = vsel_vd_vo_vd_vd(den_mask, den_out, t);
        t = vsel_vd_vo_vd_vd(neg_mask, neg_out, t);
        t = vsel_vd_vo_vd_vd(zer_mask, zer_out, t);
        t = vsel_vd_vo_vd_vd(nan_mask, nan_out, t);

        return t;
    }
#endif

    return t;
}


