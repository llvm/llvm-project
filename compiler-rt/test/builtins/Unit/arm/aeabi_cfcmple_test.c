// REQUIRES: arm-target-arch || armv6m-target-arch
// RUN: %arm_call_apsr -o %t.aspr.o
// RUN: %clang_builtins %s  %t.aspr.o %librt -o %t && %run %t

//===-- aeabi_cfcmple.c - Test __aeabi_cfcmple and __aeabi_cfrcmple -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __aeabi_cfcmple and __aeabi_cfrcmple for the compiler_rt
// library.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "call_apsr.h"

#if __arm__

extern __attribute__((pcs("aapcs"))) void __aeabi_cfcmple(float a, float b);
extern __attribute__((pcs("aapcs"))) void __aeabi_cfrcmple(float a, float b);

int test__aeabi_cfcmple(float a, float b, int expected)
{
    int32_t cpsr_value = call_apsr_f(a, b, __aeabi_cfcmple);
    int32_t r_cpsr_value = call_apsr_f(b, a, __aeabi_cfrcmple);

    if (cpsr_value != r_cpsr_value) {
        printf("error: __aeabi_cfcmple(%f, %f) != __aeabi_cfrcmple(%f, %f)\n", a, b, b, a);
        return 1;
    }

    int expected_z, expected_c;
    if (expected == -1) {
        expected_z = 0;
        expected_c = 0;
    } else if (expected == 0) {
        expected_z = 1;
        expected_c = 1;
    } else {
        // a or b is NaN, or a > b
        expected_z = 0;
        expected_c = 1;
    }

    union cpsr cpsr = { .value = cpsr_value };
    if (expected_z != cpsr.flags.z || expected_c != cpsr.flags.c) {
        printf("error in __aeabi_cfcmple(%f, %f) => (Z = %d, C = %d), expected (Z = %d, C = %d)\n",
               a, b, cpsr.flags.z, cpsr.flags.c, expected_z, expected_c);
        return 1;
    }

    cpsr.value = r_cpsr_value;
    if (expected_z != cpsr.flags.z || expected_c != cpsr.flags.c) {
        printf("error in __aeabi_cfrcmple(%f, %f) => (Z = %d, C = %d), expected (Z = %d, C = %d)\n",
               a, b, cpsr.flags.z, cpsr.flags.c, expected_z, expected_c);
        return 1;
    }
    return 0;
}
#endif

int main()
{
#if __arm__
    if (test__aeabi_cfcmple(1.0, 1.0, 0))
        return 1;
    if (test__aeabi_cfcmple(1234.567, 765.4321, 1))
        return 1;
    if (test__aeabi_cfcmple(765.4321, 1234.567, -1))
        return 1;
    if (test__aeabi_cfcmple(-123.0, -678.0, 1))
        return 1;
    if (test__aeabi_cfcmple(-678.0, -123.0, -1))
        return 1;
    if (test__aeabi_cfcmple(0.0, -0.0, 0))
        return 1;
    if (test__aeabi_cfcmple(1.0, NAN, 1))
        return 1;
    if (test__aeabi_cfcmple(NAN, 1.0, 1))
        return 1;
    if (test__aeabi_cfcmple(NAN, NAN, 1))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}
