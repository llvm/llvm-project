
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <common.h>

vfloat __attribute__((noinline)) atan2_vec(vfloat const y, vfloat const x) {

    // Helpful constants:
    const vfloat pi_3_over_4 = vcast_vf_f(PI_3_OVER_4);
    const vfloat pi_over_4 = vcast_vf_f(PI_OVER_4);
    const vfloat pi_2 = vcast_vf_f(PI_OVER_2);
    const vfloat pi = vcast_vf_f(PI);

    unsigned long long int AbsMask = 0x7FFFFFFF;
    float AbsMask_as_float = *(float *)&AbsMask;

    vfloat yAbs = vreinterpret_vf_vm(
        vand_vm_vm_vm(vreinterpret_vm_vf(y),
                      vreinterpret_vm_vf(vcast_vf_f(AbsMask_as_float))));

    vfloat xAbs = vreinterpret_vf_vm(
        vand_vm_vm_vm(vreinterpret_vm_vf(x),
                      vreinterpret_vm_vf(vcast_vf_f(AbsMask_as_float))));

    vopmask yBigger = vgt_vo_vf_vf(yAbs, xAbs);

#ifdef PERF_USE_TWO_DIVIDES
    // Different vector sizes seems to prefer different code here, most are
    // faster with using the 2 divides here, with the exception of avx512 which
    // is (noticably) faster with just the 1 divide.
    vfloat xReduced = vsel_vf_vo_vf_vf(yBigger, x / y, y / x);
#else
    // This seems to have better performance on avx512, while the other is
    // better for everything else
    vfloat xReduced =
        vsel_vf_vo_vf_vf(yBigger, x, y) / vsel_vf_vo_vf_vf(yBigger, y, x);
#endif

    // The same Estrin scheme as is used in atan(x):
    vfloat x2 = xReduced * xReduced;
    vfloat x4 = x2 * x2;
    vfloat x8 = x4 * x4;
    vfloat x16 = x8 * x8;

    // Convert our polynomial constants into vectors:
    const vfloat D1 = vcast_vf_f(C1);
    const vfloat D2 = vcast_vf_f(C2);
    const vfloat D3 = vcast_vf_f(C3);
    const vfloat D4 = vcast_vf_f(C4);
    const vfloat D5 = vcast_vf_f(C5);
    const vfloat D6 = vcast_vf_f(C6);
    const vfloat D7 = vcast_vf_f(C7);
    const vfloat D8 = vcast_vf_f(C8);

    // First layer of Estrin:
    vfloat L1 = vfma_vf_vf_vf_vf(x2, D2, D1);
    vfloat L2 = vfma_vf_vf_vf_vf(x2, D4, D3);
    vfloat L3 = vfma_vf_vf_vf_vf(x2, D6, D5);
    vfloat L4 = vfma_vf_vf_vf_vf(x2, D8, D7);

    // Second layer of estrin
    vfloat M1 = vfma_vf_vf_vf_vf(x4, L2, L1);
    vfloat M2 = vfma_vf_vf_vf_vf(x4, L4, L3);

    vfloat poly = vfma_vf_vf_vf_vf(x8, M2, M1);

    vfloat result_f = poly;

    // A vfloat that contains 0x80000000:
    unsigned long long int SignBit = 0x80000000;
    float SignBit_as_float = *(float *)&SignBit;
    vfloat SignBit_as_floatV = vcast_vf_f(SignBit_as_float);

    // pi/2 with the sign of xReduced:
    vfloat signedPi_2 = vreinterpret_vf_vm(
        vreinterpret_vm_vf(pi_2) |
        (vreinterpret_vm_vf(xReduced) & vreinterpret_vm_vf(SignBit_as_floatV)));

    vfloat pi_factor = vsel_vf_vo_vf_vf(yBigger, signedPi_2, vcast_vf_f(0.0f));

    xReduced = vsel_vf_vo_vf_vf(yBigger, -xReduced, xReduced);

    result_f = vfma_vf_vf_vf_vf(x2 * xReduced, poly, xReduced);

    // copysign(pi, y):
    const vfloat signedPi = vreinterpret_vf_vm(
        vreinterpret_vm_vf(pi) |
        (vreinterpret_vm_vf(y) & vreinterpret_vm_vf(SignBit_as_floatV)));

    // Needs to include -0.0 in the negative comparison
    // We do this by cast to an int and comparing to -1:
    vint2 pi_factor_as_vmask =
        (~vgt_vi2_vi2_vi2(vreinterpret_vi2_vf(x), vcast_vi2_i(-1))) &
        vreinterpret_vi2_vf(signedPi);

    pi_factor += vreinterpret_vf_vi2(pi_factor_as_vmask);

    result_f += pi_factor;

    // We need to mask off some special values, mainly infinities and 0's
    // Fortunately for all cases we always have (xAbs == yAbs)
    // Get special return value:
    vfloat special_return_value = vsel_vf_vo_vf_vf(
        vlt_vo_vf_vf(x, vcast_vf_f(0.0f)), pi_3_over_4, pi_over_4);

    // Special return values for (y, x) = (+-0, +-0)
    vint2 special_yx_zero_zero_as_int =
        ~vgt_vi2_vi2_vi2(vreinterpret_vi2_vf(x), vcast_vi2_i(-1)) &
        vreinterpret_vi2_vf(pi);

    vfloat special_yx_zero_zero =
        vreinterpret_vf_vi2(special_yx_zero_zero_as_int);

    // Check for (y, x) = (+-0.0, +-0.0)
    special_return_value =
        vsel_vf_vo_vf_vf(veq_vo_vf_vf(x, vcast_vf_f(0.0f)),
                         special_yx_zero_zero, special_return_value);

    result_f = vsel_vf_vo_vf_vf(veq_vo_vf_vf(yAbs, xAbs), special_return_value,
                                result_f);

    // copysign(result_f, y):
    result_f = vreinterpret_vf_vm(
        vreinterpret_vm_vf(result_f) |
        (vreinterpret_vm_vf(y) & vreinterpret_vm_vf(SignBit_as_floatV)));

    return result_f;
}
