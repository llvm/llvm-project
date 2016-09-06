//===-- aeabi_ldivmod.S - EABI ldivmod implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

// struct { int64_t quot, int64_t rem}
//        __aeabi_ldivmod(int64_t numerator, int64_t denominator) {
//   int64_t rem, quot;
//   quot = __divmoddi4(numerator, denominator, &rem);
//   return {quot, rem};
// }

        .syntax unified
        .p2align 2
DEFINE_COMPILERRT_FUNCTION(__aeabi_ldivmod)
        push    {r11, lr}
        sub     sp, sp, #16
        add     r12, sp, #8
        str     r12, [sp]
        bl      SYMBOL_NAME(__divmoddi4)
        ldr     r2, [sp, #8]
        ldr     r3, [sp, #12]
        add     sp, sp, #16
        pop     {r11, pc}
END_COMPILERRT_FUNCTION(__aeabi_ldivmod)

NO_EXEC_STACK_DIRECTIVE

