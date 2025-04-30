
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <common.h>

vdouble __attribute__((noinline))
atan2_d_vec(vdouble const y, vdouble const x) {
    // Helpful constants:
    const vdouble pi_3_over_4 = vcast_vd_d(PI_3_OVER_4);
    const vdouble pi_over_4 = vcast_vd_d(PI_OVER_4);
    const vdouble pi_2 = vcast_vd_d(PI_OVER_2);
    const vdouble pi = vcast_vd_d(PI);

    unsigned long long int AbsMask = 0x7FFFFFFFFFFFFFFF;
    double AbsMask_as_double = *(double *)&AbsMask;

    vdouble yAbs = vreinterpret_vd_vm(
        vand_vm_vm_vm(vreinterpret_vm_vd(y),
                      vreinterpret_vm_vd(vcast_vd_d(AbsMask_as_double))));

    vdouble xAbs = vreinterpret_vd_vm(
        vand_vm_vm_vm(vreinterpret_vm_vd(x),
                      vreinterpret_vm_vd(vcast_vd_d(AbsMask_as_double))));

    // We need to check if x is negative, but include -0.0 in this.
    // We do this by casting to int and 2's complement:
    vmask xNegative = (vreinterpret_vm_vd(x) < 0);

    vopmask yBigger = vgt_vo_vd_vd(yAbs, xAbs);

#ifdef PERF_USE_TWO_DIVIDES
    // Different vector sizes seems to prefer different code here, most are
    // faster with using the 2 divides here, with the exception of avx512 which
    // is (noticably) faster with just the 1 divide.
    vdouble xReduced = vsel_vd_vo_vd_vd(yBigger, x / y, y / x);
#else
    // This seems to have better performance on avx512, while the other is
    // better for everything else
    vdouble xReduced =
        vsel_vd_vo_vd_vd(yBigger, x, y) / vsel_vd_vo_vd_vd(yBigger, y, x);
#endif

    // The same Estrin scheme as is used in atan(x):
    vdouble x2 = xReduced * xReduced;
    vdouble x4 = x2 * x2;
    vdouble x8 = x4 * x4;
    vdouble x16 = x8 * x8;

    // Convert our polynomial constants into vectors:
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

    // The same Estrin scheme as is used in atan(x):
    vdouble L1 = vfma_vd_vd_vd_vd(x2, D6, D5);
    vdouble L2 = vfma_vd_vd_vd_vd(x2, D8, D7);
    vdouble L3 = vfma_vd_vd_vd_vd(x2, D10, D9);
    vdouble L4 = vfma_vd_vd_vd_vd(x2, D12, D11);
    vdouble L5 = vfma_vd_vd_vd_vd(x2, D14, D13);
    vdouble L6 = vfma_vd_vd_vd_vd(x2, D16, D15);
    vdouble L7 = vfma_vd_vd_vd_vd(x2, D18, D17);
    vdouble L8 = vfma_vd_vd_vd_vd(x2, D20, D19);

    // L1 + x4*L2 + x8*L3 + x12*L4 + x16*L5 + x20*L6 + x24*L7 + x28*L8
    vdouble M1 = vfma_vd_vd_vd_vd(x4, L2, L1);
    vdouble M2 = vfma_vd_vd_vd_vd(x4, L4, L3);
    vdouble M3 = vfma_vd_vd_vd_vd(x4, L6, L5);
    vdouble M4 = vfma_vd_vd_vd_vd(x4, L8, L7);

    // M1 + x8*M2 + x16*M3 + x24*M4
    // (M1 + x8*M2) + x16*(M3 + x8*M4)
    vdouble N1 = vfma_vd_vd_vd_vd(x8, M2, M1);
    vdouble N2 = vfma_vd_vd_vd_vd(x8, M4, M3);

    // D2 + x2*D3 + x4*D4 + x6*(N1 + x16*N2):
    vdouble poly = vfma_vd_vd_vd_vd(x16, N2, N1);

    poly = vfma_vd_vd_vd_vd(x4, vfma_vd_vd_vd_vd(x2, poly, D4),
                            vfma_vd_vd_vd_vd(x2, D3, D2));

    vdouble result_d = poly;

    // A vdouble that contains 0x8000000000000000:
    unsigned long long int SignBit = 0x8000000000000000;
    double SignBit_as_double = *(double *)&SignBit;
    vdouble SignBit_as_doubleV = vcast_vd_d(SignBit_as_double);

    // pi/2 with the sign of xReduced:
    vdouble signedPi_2 = vreinterpret_vd_vm(
        vreinterpret_vm_vd(pi_2) | (vreinterpret_vm_vd(xReduced) &
                                    vreinterpret_vm_vd(SignBit_as_doubleV)));

    vdouble pi_factor = vsel_vd_vo_vd_vd(yBigger, signedPi_2, vcast_vd_d(0.0));

    xReduced = vsel_vd_vo_vd_vd(yBigger, -xReduced, xReduced);

    result_d = vfma_vd_vd_vd_vd(x2 * xReduced, poly, xReduced);

    // pi with the sign of y:
    const vdouble signedPi = vreinterpret_vd_vm(
        vreinterpret_vm_vd(pi) |
        (vreinterpret_vm_vd(y) & vreinterpret_vm_vd(SignBit_as_doubleV)));

    pi_factor += vreinterpret_vd_vm(xNegative & vreinterpret_vm_vd(signedPi));

    result_d += pi_factor;

    // We need to mask off some special values, mainly infinities and 0's
    // Fortunately for all cases we always have (xAbs == yAbs)
    // Get special return value:
    vdouble special_return_value = vsel_vd_vo_vd_vd(
        vlt_vo_vd_vd(x, vcast_vd_d(0.0f)), pi_3_over_4, pi_over_4);

    vdouble special_yx_zero_zero =
        vreinterpret_vd_vm(xNegative & vreinterpret_vm_vd(pi));

    // Check for (y, x) = (+-0.0, +-0.0)
    special_return_value =
        vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0.0)), special_yx_zero_zero,
                         special_return_value);

    result_d = vsel_vd_vo_vd_vd(veq_vo_vd_vd(yAbs, xAbs), special_return_value,
                                result_d);

    // copysign(special_return_value, y):
    result_d = vreinterpret_vd_vm(
        vreinterpret_vm_vd(result_d) |
        (vreinterpret_vm_vd(y) & vreinterpret_vm_vd(SignBit_as_doubleV)));

    return result_d;
}

