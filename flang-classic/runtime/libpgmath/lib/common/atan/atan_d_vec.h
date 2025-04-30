
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <common.h>

vdouble __attribute__((noinline)) atan_d_vec(vdouble const x) {

    unsigned long long int AbsMask = 0x7FFFFFFFFFFFFFFF;
    double AbsMask_as_double = *(double *)&AbsMask;

    vdouble f_abs = vreinterpret_vd_vm(
        vand_vm_vm_vm(vreinterpret_vm_vd(x),
                      vreinterpret_vm_vd(vcast_vd_d(AbsMask_as_double))));
    vdouble ans_sgn = vreinterpret_vd_vm(
        vxor_vm_vm_vm(vreinterpret_vm_vd(f_abs), vreinterpret_vm_vd(x)));

    vopmask f_big = vgt_vo_vd_vd(f_abs, vcast_vd_d(1.0));

    vdouble xReduced = vsel_vd_vo_vd_vd(f_big, 1.0/x, x);

    vdouble x2 = xReduced * xReduced;
    vdouble x4 = x2 * x2;
    vdouble x8 = x4 * x4;
    vdouble x16 = x8 * x8;
    
    //Convert our polynomial constants into vectors:
    const vdouble D2 = vcast_vd_d(C2);
    const vdouble D3 = vcast_vd_d(C3);
    const vdouble D4 = vcast_vd_d(C4);
    const vdouble D5 = vcast_vd_d(C5);
    const vdouble D6 = vcast_vd_d(C6);
    const vdouble D7 = vcast_vd_d(C7);
    const vdouble D8 = vcast_vd_d(C8);
    const vdouble D9 = vcast_vd_d(C9);
    const vdouble D10 = vcast_vd_d(C10);
    const vdouble D11 = vcast_vd_d(C11);
    const vdouble D12 = vcast_vd_d(C12);
    const vdouble D13 = vcast_vd_d(C13);
    const vdouble D14 = vcast_vd_d(C14);
    const vdouble D15 = vcast_vd_d(C15);
    const vdouble D16 = vcast_vd_d(C16);
    const vdouble D17 = vcast_vd_d(C17);
    const vdouble D18 = vcast_vd_d(C18);
    const vdouble D19 = vcast_vd_d(C19);
    const vdouble D20 = vcast_vd_d(C20);

    // Estrin:
    // We want D2 + x2*(D3 + x2*(D4 + (.....))) = D2 + x2*D3 + x4*D4 + x6*D5 +
    // ... + x36 * D20

    // First layer of Estrin
    vdouble L1 = vfma_vd_vd_vd_vd(x2, D3, D2);
    vdouble L2 = vfma_vd_vd_vd_vd(x2, D5, D4);
    vdouble L3 = vfma_vd_vd_vd_vd(x2, D7, D6);
    vdouble L4 = vfma_vd_vd_vd_vd(x2, D9, D8);
    vdouble L5 = vfma_vd_vd_vd_vd(x2, D11, D10);
    vdouble L6 = vfma_vd_vd_vd_vd(x2, D13, D12);
    vdouble L7 = vfma_vd_vd_vd_vd(x2, D15, D14);
    vdouble L8 = vfma_vd_vd_vd_vd(x2, D17, D16);
    vdouble L9 = vfma_vd_vd_vd_vd(x2, D19, D18);

    // We now want:
    //  L1 + x4*L2 + x8*L3 + x12*L4 + x16*L5 + x20*L6 + x24*L7 + x28*L8 + x32*L9
    //  + x36*C20
    // (L1 + x4*L2) + x8*(L3 + x4*L4) + x16*(L5 + x4*L6) + x24*(L7 + x4*L8) +
    // x32(*L9 + x4*C20)

    // Second layer of Estrin
    vdouble M1 = vfma_vd_vd_vd_vd(x4, L2, L1);
    vdouble M2 = vfma_vd_vd_vd_vd(x4, L4, L3);
    vdouble M3 = vfma_vd_vd_vd_vd(x4, L6, L5);
    vdouble M4 = vfma_vd_vd_vd_vd(x4, L8, L7);
    vdouble M5 = vfma_vd_vd_vd_vd(x4, D20, L9);

    // We now want:
    // M1 + x8*M2 + x16*M3 + x24*M4 + x32*M5
    // (M1 + x8*M2) + x16*(M3 + x8*M4 + x16*M5)
    vdouble N1 = vfma_vd_vd_vd_vd(x8, M2, M1);
    vdouble N2 = vfma_vd_vd_vd_vd(x16, M5, M3 + x8 * M4);

    vdouble poly = vfma_vd_vd_vd_vd(x16, N2, N1);

    //This is a copysign(pi/2, x);
    const vdouble signedPi_2 = vreinterpret_vd_vm(vor_vm_vm_vm(
        vreinterpret_vm_vd(vcast_vd_d(PI_2)),
        vreinterpret_vm_vd(ans_sgn)));

    vdouble result_f_big     = vfma_vd_vd_vd_vd( -x2 * xReduced, poly, signedPi_2 - xReduced);
    vdouble result_not_f_big = vfma_vd_vd_vd_vd(x2 * xReduced, poly, xReduced);
    
    vdouble result = vsel_vd_vo_vd_vd(f_big, result_f_big, result_not_f_big);

    result = vreinterpret_vd_vm(vreinterpret_vm_vd(result) | vreinterpret_vm_vd(ans_sgn));

    return result;
}
