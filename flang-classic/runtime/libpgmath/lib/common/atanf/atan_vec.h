
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <common.h>

vfloat __attribute__((noinline)) atan_vec(vfloat const x) {

    // Vectorise our coefficients:
    vfloat D1 = vcast_vf_f(C1);
    vfloat D2 = vcast_vf_f(C2);
    vfloat D3 = vcast_vf_f(C3);
    vfloat D4 = vcast_vf_f(C4);
    vfloat D5 = vcast_vf_f(C5);
    vfloat D6 = vcast_vf_f(C6);
    vfloat D7 = vcast_vf_f(C7);
    vfloat D8 = vcast_vf_f(C8);

    unsigned int AbsMask = 0x7FFFFFFF;
    float AbsMask_as_float = *(float *)&AbsMask;

    vfloat xAbs = vreinterpret_vf_vm(
        vand_vm_vm_vm(vreinterpret_vm_vf(x),
                      vreinterpret_vm_vf(vcast_vf_f(AbsMask_as_float))));

    vfloat ans_sgn = vreinterpret_vf_vm(
        vxor_vm_vm_vm(vreinterpret_vm_vf(xAbs), vreinterpret_vm_vf(x)));

    vfloat xReduced = x;

    vopmask x_big = vgt_vo_vf_vf(xAbs, vcast_vf_f(1.0f));

    xReduced = vsel_vf_vo_vf_vf(x_big, vcast_vf_f(1.0f) / xReduced, xReduced);

    vfloat x2 = xReduced * xReduced;
    vfloat x4 = x2 * x2;
    vfloat x8 = x4 * x4;

    // First layer of Estrin:
    vfloat L1 = vfma_vf_vf_vf_vf(x2, D2, D1);
    vfloat L2 = vfma_vf_vf_vf_vf(x2, D4, D3);
    vfloat L3 = vfma_vf_vf_vf_vf(x2, D6, D5);
    vfloat L4 = vfma_vf_vf_vf_vf(x2, D8, D7);

    // Second layer of estrin
    vfloat M1 = vfma_vf_vf_vf_vf(x4, L2, L1);
    vfloat M2 = vfma_vf_vf_vf_vf(x4, L4, L3);

    vfloat poly = vfma_vf_vf_vf_vf(x8, M2, M1);

    // copysign(pi/2, x):
    const vfloat signedPi_2 = vreinterpret_vf_vm(vor_vm_vm_vm(
        vreinterpret_vm_vf(vcast_vf_f(PI_2)), vreinterpret_vm_vf(ans_sgn)));

    vfloat result_x_big =
        vfma_vf_vf_vf_vf(-x2 * xReduced, poly, signedPi_2 - xReduced);
    vfloat result_not_x_big = vfma_vf_vf_vf_vf(x2 * xReduced, poly, xReduced);

    vfloat result = vsel_vf_vo_vf_vf(x_big, result_x_big, result_not_x_big);

    //Make sure atanf(-0.0f) = -0.0f:
    result = vreinterpret_vf_vm(vor_vm_vm_vm(vreinterpret_vm_vf(result), vreinterpret_vm_vf(ans_sgn)));

    return result;
}
