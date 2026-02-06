//===-- aeabi_uread4.S - ARM EABI Helper — Unaligned 4-Byte Memory Read ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements __aeabi_uread4 for unaligned memory accesses.
// Reference: Arm RTABI32 Specification.
// https://github.com/ARM-software/abi-aa/blob/main/rtabi32/rtabi32.rst#unaligned-memory-access
//===-------------------------------------------------------------------------------------===//

#include "../assembly.h"

        .syntax unified
        .p2align 2
DEFINE_COMPILERRT_FUNCTION(__aeabi_uread4)
#if defined(__thumb__) && !defined(__thumb2__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        ldrb    r1, [r0]
        ldrb    r2, [r0, #1]
        lsls    r2, r2, #8
        adds    r1, r2, r1
        ldrb    r2, [r0, #2]
        lsls    r2, r2, #16
        ldrb    r0, [r0, #3]
        lsls    r0, r0, #24
        adds    r0, r0, r2
        adds    r0, r0, r1
#else
        ldrb    r1, [r0, #3]
        ldrb    r2, [r0, #2]
        lsls    r2, r2, #8
        adds    r1, r2, r1
        ldrb    r2, [r0, #1]
        lsls    r2, r2, #16
        ldrb    r0, [r0]
        lsls    r0, r0, #24
        adds    r0, r0, r2
        adds    r0, r0, r1
#endif
        bx lr
#else
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        ldrb    r1, [r0]
        ldrb    r2, [r0, #1]
        ldrb    r3, [r0, #2]
        ldrb    r0, [r0, #3]
        orr     r0, r3, r0, lsl #8
        orr     r1, r1, r2, lsl #8
        orr     r0, r1, r0, lsl #16
#else
        ldrb    r1, [r0]
        ldrb    r2, [r0, #1]
        ldrb    r3, [r0, #2]
        ldrb    r0, [r0, #3]
        orr     r1, r2, r1, lsl #8
        orr     r0, r0, r3, lsl #8
        orr     r0, r0, r1, lsl #16
#endif
        bx      lr
#endif
END_COMPILERRT_FUNCTION(__aeabi_uread4)

