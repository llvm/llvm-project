//===-- aeabi_uwrite8.S - ARM EABI Helper — Unaligned 8-Byte Memory Write--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements __aeabi_uwrite8 for unaligned memory accesses.
// Reference: Arm RTABI32 Specification.
// https://github.com/ARM-software/abi-aa/blob/main/rtabi32/rtabi32.rst#unaligned-memory-access
//===-------------------------------------------------------------------------------------===//

#include "../assembly.h"

        .syntax unified
        .p2align 2
DEFINE_COMPILERRT_FUNCTION(__aeabi_uwrite8)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        strb    r0, [r2, #0]
        lsrs    r3, r0, #8
        strb    r3, [r2, #1]
        lsrs    r3, r0, #16
        strb    r3, [r2, #2]
        lsrs    r3, r0, #24
        strb    r3, [r2, #3]

        strb    r1, [r2, #4]
        lsrs    r3, r1, #8
        strb    r3, [r2, #5]
        lsrs    r3, r1, #16
        strb    r3, [r2, #6]
        lsrs    r3, r1, #24
        strb    r3, [r2, #7]
#else
        lsrs    r3, r0, #24
        strb    r3, [r2, #0]
        lsrs    r3, r0, #16
        strb    r3, [r2, #1]
        lsrs    r3, r0, #8
        strb    r3, [r2, #2]
        strb    r0, [r2, #3]

        lsrs    r3, r1, #24
        strb    r3, [r2, #4]
        lsrs    r3, r1, #16
        strb    r3, [r2, #5]
        lsrs    r3, r1, #8
        strb    r3, [r2, #6]
        strb    r1, [r2, #7]
#endif
        bx      lr
END_COMPILERRT_FUNCTION(__aeabi_uwrite8)
