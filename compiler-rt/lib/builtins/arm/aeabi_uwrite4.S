//===-- aeabi_uwrite4.S - ARM EABI Helper — Unaligned 4-Byte Memory Write--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements __aeabi_uwrite4 for unaligned memory accesses.
// Reference: Arm RTABI32 Specification.
// https://github.com/ARM-software/abi-aa/blob/main/rtabi32/rtabi32.rst#unaligned-memory-access
//===-------------------------------------------------------------------------------------===//

#include "../assembly.h"

        .syntax unified
        .p2align 2
DEFINE_COMPILERRT_FUNCTION(__aeabi_uwrite4)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        lsrs    r2, r0, #24
        strb    r0, [r1]
        strb    r2, [r1, #3]
        lsrs    r2, r0, #16
        strb    r2, [r1, #2]
        lsrs    r2, r0, #8
        strb    r2, [r1, #1]
#else
        lsrs    r2, r0, #8
        strb    r0, [r1, #3]
        strb    r2, [r1, #2]
        lsrs    r2, r0, #16
        strb    r2, [r1, #1]
        lsrs    r2, r0, #24
        strb    r2, [r1]
#endif
        bx      lr
END_COMPILERRT_FUNCTION(__aeabi_uwrite4)
