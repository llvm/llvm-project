// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld -fix-cortex-a53-843419 -z separate-code -verbose %t.o -o %t2 2>&1 | FileCheck -check-prefix CHECK-PRINT %s
// RUN: llvm-objdump --no-print-imm-hex --triple=aarch64-linux-gnu -d %t2 | FileCheck %s --check-prefixes=CHECK,CHECK-FIX
// RUN: ld.lld %t.o -z separate-code -o %t3
// RUN: llvm-objdump --no-print-imm-hex --triple=aarch64-linux-gnu -d %t3 | FileCheck %s --check-prefixes=CHECK,CHECK-NOFIX
// RUN: ld.lld -fix-cortex-a53-843419 -r -z separate-code %t.o -o %t4
// RUN: llvm-objdump --no-print-imm-hex --triple=aarch64-linux-gnu -d %t4 | FileCheck %s --check-prefixes=CHECK-RELOCATABLE
// Test cases for Cortex-A53 Erratum 843419
// See ARM-EPM-048406 Cortex_A53_MPCore_Software_Developers_Errata_Notice.pdf
// for full erratum details.
// In Summary
// 1.)
// ADRP (0xff8 or 0xffc).
// 2.)
// - load or store single register or either integer or vector registers.
// - STP or STNP of either vector or vector registers.
// - Advanced SIMD ST1 store instruction.
// - Must not write Rn.
// 3.) optional instruction, can't be a branch, must not write Rn, may read Rn.
// 4.) A load or store instruction from the Load/Store register unsigned
// immediate class using Rn as the base register.

// Each section contains a sequence of instructions that should be recognized
// as erratum 843419. The test cases cover the major variations such as:
// - adrp starts at 0xfff8 or 0xfffc.
// - Variations in instruction class for instruction 2.
// - Optional instruction 3 present or not.
// - Load or store for instruction 4.

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 211FF8 in unpatched output.
// CHECK: <t3_ff8_ldr>:
// CHECK-NEXT:    211ff8:        f0000260        adrp    x0, 0x260000
// CHECK-NEXT:    211ffc:        f9400021        ldr             x1, [x1]
// CHECK-FIX:     212000:        1400c803        b       0x24400c
// CHECK-NOFIX:   212000:        f9400000        ldr             x0, [x0]
// CHECK-NEXT:    212004:        d65f03c0        ret
// CHECK-RELOCATABLE: <t3_ff8_ldr>:
// CHECK-RELOCATABLE-NEXT:    ff8:        90000000        adrp    x0, 0x0
// CHECK-RELOCATABLE-NEXT:    ffc:        f9400021        ldr             x1, [x1]
// CHECK-RELOCATABLE-NEXT:   1000:        f9400000        ldr             x0, [x0]
// CHECK-RELOCATABLE-NEXT:   1004:        d65f03c0        ret

        .section .text.01, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldr
        .type t3_ff8_ldr, %function
        .space 4096 - 8
t3_ff8_ldr:
        adrp x0, dat1
        ldr x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 213FF8 in unpatched output.
// CHECK: <t3_ff8_ldrsimd>:
// CHECK-NEXT:    213ff8:        b0000260        adrp    x0, 0x260000
// CHECK-NEXT:    213ffc:        bd400021        ldr             s1, [x1]
// CHECK-FIX:     214000:        1400c005        b       0x244014
// CHECK-NOFIX:   214000:        f9400402        ldr     x2, [x0, #8]
// CHECK-NEXT:    214004:        d65f03c0        ret
        .section .text.02, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldrsimd
        .type t3_ff8_ldrsimd, %function
        .space 4096 - 8
t3_ff8_ldrsimd:
        adrp x0, dat2
        ldr s1, [x1, #0]
        ldr x2, [x0, :got_lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 215FFC in unpatched output.
// CHECK: <t3_ffc_ldrpost>:
// CHECK-NEXT:    215ffc:        f0000240        adrp    x0, 0x260000
// CHECK-NEXT:    216000:        bc408421        ldr     s1, [x1], #8
// CHECK-FIX:     216004:        1400b806        b       0x24401c
// CHECK-NOFIX:   216004:        f9400803        ldr     x3, [x0, #16]
// CHECK-NEXT:    216008:        d65f03c0        ret
        .section .text.03, "ax", %progbits
        .balign 4096
        .globl t3_ffc_ldrpost
        .type t3_ffc_ldrpost, %function
        .space 4096 - 4
t3_ffc_ldrpost:
        adrp x0, dat3
        ldr s1, [x1], #8
        ldr x3, [x0, :got_lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 217FF8 in unpatched output.
// CHECK: <t3_ff8_strpre>:
// CHECK-NEXT:    217ff8:        b0000240        adrp    x0, 0x260000
// CHECK-NEXT:    217ffc:        bc008c21        str     s1, [x1, #8]!
// CHECK-FIX:     218000:        1400b009        b       0x244024
// CHECK-NOFIX:   218000:        f9400c02        ldr     x2, [x0, #24]
// CHECK-NEXT:    218004:        d65f03c0        ret
        .section .text.04, "ax", %progbits
        .balign 4096
        .globl t3_ff8_strpre
        .type t3_ff8_strpre, %function
        .space 4096 - 8
t3_ff8_strpre:
        adrp x0, dat1
        str s1, [x1, #8]!
        ldr x2, [x0, :lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 219FFC in unpatched output.
// CHECK: <t3_ffc_str>:
// CHECK-NEXT:    219ffc:        f000023c        adrp    x28, 0x260000
// CHECK-NEXT:    21a000:        f9000042        str             x2, [x2]
// CHECK-FIX:     21a004:        1400a80a        b       0x24402c
// CHECK-NOFIX:   21a004:        f900139c        str     x28, [x28, #32]
// CHECK-NEXT:    21a008:        d65f03c0        ret
        .section .text.05, "ax", %progbits
        .balign 4096
        .globl t3_ffc_str
        .type t3_ffc_str, %function
        .space 4096 - 4
t3_ffc_str:
        adrp x28, dat2
        str x2, [x2, #0]
        str x28, [x28, :lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 21BFFC in unpatched output.
// CHECK: <t3_ffc_strsimd>:
// CHECK-NEXT:    21bffc:        b000023c        adrp    x28, 0x260000
// CHECK-NEXT:    21c000:        b9000044        str             w4, [x2]
// CHECK-FIX:     21c004:        1400a00c        b       0x244034
// CHECK-NOFIX:   21c004:        f9001784        str     x4, [x28, #40]
// CHECK-NEXT:    21c008:        d65f03c0        ret
        .section .text.06, "ax", %progbits
        .balign 4096
        .globl t3_ffc_strsimd
        .type t3_ffc_strsimd, %function
        .space 4096 - 4
t3_ffc_strsimd:
        adrp x28, dat3
        str w4, [x2, #0]
        str x4, [x28, :lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 21DFF8 in unpatched output.
// CHECK: <t3_ff8_ldrunpriv>:
// CHECK-NEXT:    21dff8:        f000021d        adrp    x29, 0x260000
// CHECK-NEXT:    21dffc:        38400841        ldtrb           w1, [x2]
// CHECK-FIX:     21e000:        1400980f        b       0x24403c
// CHECK-NOFIX:   21e000:        f94003bd        ldr             x29, [x29]
// CHECK-NEXT:    21e004:        d65f03c0        ret
        .section .text.07, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldrunpriv
        .type t3_ff8_ldrunpriv, %function
        .space 4096 - 8
t3_ff8_ldrunpriv:
        adrp x29, dat1
        ldtrb w1, [x2, #0]
        ldr x29, [x29, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 21FFFC in unpatched output.
// CHECK: <t3_ffc_ldur>:
// CHECK-NEXT:    21fffc:        b000021d        adrp    x29, 0x260000
// CHECK-NEXT:    220000:        b8404042        ldur    w2, [x2, #4]
// CHECK-FIX:     220004:        14009010        b       0x244044
// CHECK-NOFIX:   220004:        f94007bd        ldr     x29, [x29, #8]
// CHECK-NEXT:    220008:        d65f03c0        ret
        .balign 4096
        .globl t3_ffc_ldur
        .type t3_ffc_ldur, %function
        .space 4096 - 4
t3_ffc_ldur:
        adrp x29, dat2
        ldur w2, [x2, #4]
        ldr x29, [x29, :got_lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 221FFC in unpatched output.
// CHECK: <t3_ffc_sturh>:
// CHECK-NEXT:    221ffc:        f00001f2        adrp    x18, 0x260000
// CHECK-NEXT:    222000:        78004043        sturh   w3, [x2, #4]
// CHECK-FIX:     222004:        14008812        b       0x24404c
// CHECK-NOFIX:   222004:        f9400a41        ldr     x1, [x18, #16]
// CHECK-NEXT:    222008:        d65f03c0        ret
        .section .text.09, "ax", %progbits
        .balign 4096
        .globl t3_ffc_sturh
        .type t3_ffc_sturh, %function
        .space 4096 - 4
t3_ffc_sturh:
        adrp x18, dat3
        sturh w3, [x2, #4]
        ldr x1, [x18, :got_lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 223FF8 in unpatched output.
// CHECK: <t3_ff8_literal>:
// CHECK-NEXT:    223ff8:        b00001f2        adrp    x18, 0x260000
// CHECK-NEXT:    223ffc:        58ffffe3        ldr     x3, 0x223ff8
// CHECK-FIX:     224000:        14008015        b       0x244054
// CHECK-NOFIX:   224000:        f9400e52        ldr     x18, [x18, #24]
// CHECK-NEXT:    224004:        d65f03c0        ret
        .section .text.10, "ax", %progbits
        .balign 4096
        .globl t3_ff8_literal
        .type t3_ff8_literal, %function
        .space 4096 - 8
t3_ff8_literal:
        adrp x18, dat1
        ldr x3, t3_ff8_literal
        ldr x18, [x18, :lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 225FFC in unpatched output.
// CHECK: <t3_ffc_register>:
// CHECK-NEXT:    225ffc:        f00001cf        adrp    x15, 0x260000
// CHECK-NEXT:    226000:        f8616843        ldr             x3, [x2, x1]
// CHECK-FIX:     226004:        14007816        b       0x24405c
// CHECK-NOFIX:   226004:        f94011ea        ldr     x10, [x15, #32]
// CHECK-NEXT:    226008:        d65f03c0        ret
        .section .text.11, "ax", %progbits
        .balign 4096
        .globl t3_ffc_register
        .type t3_ffc_register, %function
        .space 4096 - 4
t3_ffc_register:
        adrp x15, dat2
        ldr x3, [x2, x1]
        ldr x10, [x15, :lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 227FF8 in unpatched output.
// CHECK: <t3_ff8_stp>:
// CHECK-NEXT:    227ff8:        b00001d0        adrp    x16, 0x260000
// CHECK-NEXT:    227ffc:        a9000861        stp             x1, x2, [x3]
// CHECK-FIX:     228000:        14007019        b       0x244064
// CHECK-NOFIX:   228000:        f940160d        ldr     x13, [x16, #40]
// CHECK-NEXT:    228004:        d65f03c0        ret
        .section .text.12, "ax", %progbits
        .balign 4096
        .globl t3_ff8_stp
        .type t3_ff8_stp, %function
        .space 4096 - 8
t3_ff8_stp:
        adrp x16, dat3
        stp x1,x2, [x3, #0]
        ldr x13, [x16, :lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 229FFC in unpatched output.
// CHECK: <t3_ffc_stnp>:
// CHECK-NEXT:    229ffc:        f00001a7        adrp    x7, 0x260000
// CHECK-NEXT:    22a000:        a8000861        stnp            x1, x2, [x3]
// CHECK-FIX:     22a004:        1400681a        b       0x24406c
// CHECK-NOFIX:   22a004:        f9400ce9        ldr             x9, [x7, #24]
// CHECK-NEXT:    22a008:        d65f03c0        ret
        .section .text.13, "ax", %progbits
        .balign 4096
        .globl t3_ffc_stnp
        .type t3_ffc_stnp, %function
        .space 4096 - 4
t3_ffc_stnp:
        adrp x7, dat1
        stnp x1,x2, [x3, #0]
        ldr x9, [x7, :lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 22BFFC in unpatched output.
// CHECK: <t3_ffc_st1singlepost>:
// CHECK-NEXT:    22bffc:        b00001b7        adrp    x23, 0x260000
// CHECK-NEXT:    22c000:        0d820420        st1 { v0.b }[1], [x1], x2
// CHECK-FIX:     22c004:        1400601c        b       0x244074
// CHECK-NOFIX:   22c004:        f94012f6        ldr     x22, [x23, #32]
// CHECK-NEXT:    22c008:        d65f03c0        ret
        .section .text.14, "ax", %progbits
        .balign 4096
        .globl t3_ffc_st1singlepost
        .type t3_ffc_st1singlepost, %function
        .space 4096 - 4
t3_ffc_st1singlepost:
        adrp x23, dat2
        st1 { v0.b }[1], [x1], x2
        ldr x22, [x23, :lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 22DFF8 in unpatched output.
// CHECK: <t3_ff8_st1multiple>:
// CHECK-NEXT:    22dff8:        f0000197        adrp    x23, 0x260000
// CHECK-NEXT:    22dffc:        4c00a020        st1     { v0.16b, v1.16b }, [x1]
// CHECK-FIX:     22e000:        1400581f        b       0x24407c
// CHECK-NOFIX:   22e000:        f94016f8        ldr     x24, [x23, #40]
// CHECK-NEXT:    22e004:        d65f03c0        ret
        .section .text.15, "ax", %progbits
        .balign 4096
        .globl t3_ff8_st1multiple
        .type t3_ff8_st1muliple, %function
        .space 4096 - 8
t3_ff8_st1multiple:
        adrp x23, dat3
        st1 { v0.16b, v1.16b }, [x1]
        ldr x24, [x23, :lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 22FFF8 in unpatched output.
// CHECK: <t4_ff8_ldr>:
// CHECK-NEXT:    22fff8:        b0000180        adrp    x0, 0x260000
// CHECK-NEXT:    22fffc:        f9400021        ldr             x1, [x1]
// CHECK-NEXT:    230000:        8b000042        add             x2, x2, x0
// CHECK-FIX:     230004:        14005020        b       0x244084
// CHECK-NOFIX:   230004:        f9400002        ldr             x2, [x0]
// CHECK-NEXT:    230008:        d65f03c0        ret
        .section .text.16, "ax", %progbits
        .balign 4096
        .globl t4_ff8_ldr
        .type t4_ff8_ldr, %function
        .space 4096 - 8
t4_ff8_ldr:
        adrp x0, dat1
        ldr x1, [x1, #0]
        add x2, x2, x0
        ldr x2, [x0, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 231FFC in unpatched output.
// CHECK: <t4_ffc_str>:
// CHECK-NEXT:    231ffc:        f000017c        adrp    x28, 0x260000
// CHECK-NEXT:    232000:        f9000042        str             x2, [x2]
// CHECK-NEXT:    232004:        cb020020        sub             x0, x1, x2
// CHECK-FIX:     232008:        14004821        b       0x24408c
// CHECK-NOFIX:   232008:        f900079b        str     x27, [x28, #8]
// CHECK-NEXT:    23200c:        d65f03c0        ret
        .section .text.17, "ax", %progbits
        .balign 4096
        .globl t4_ffc_str
        .type t4_ffc_str, %function
        .space 4096 - 4
t4_ffc_str:
        adrp x28, dat2
        str x2, [x2, #0]
        sub x0, x1, x2
        str x27, [x28, :got_lo12:dat2]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 233FF8 in unpatched output.
// CHECK: <t4_ff8_stp>:
// CHECK-NEXT:    233ff8:        b0000170        adrp    x16, 0x260000
// CHECK-NEXT:    233ffc:        a9000861        stp             x1, x2, [x3]
// CHECK-NEXT:    234000:        9b107e03        mul             x3, x16, x16
// CHECK-FIX:     234004:        14004024        b       0x244094
// CHECK-NOFIX:   234004:        f9400a0e        ldr     x14, [x16, #16]
// CHECK-NEXT:    234008:        d65f03c0        ret
        .section .text.18, "ax", %progbits
        .balign 4096
        .globl t4_ff8_stp
        .type t4_ff8_stp, %function
        .space 4096 - 8
t4_ff8_stp:
        adrp x16, dat3
        stp x1,x2, [x3, #0]
        mul x3, x16, x16
        ldr x14, [x16, :got_lo12:dat3]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 235FF8 in unpatched output.
// CHECK: <t4_ff8_stppre>:
// CHECK-NEXT:    235ff8:        f0000150        adrp    x16, 0x260000
// CHECK-NEXT:    235ffc:        a9810861        stp     x1, x2, [x3, #16]!
// CHECK-NEXT:    236000:        9b107e03        mul             x3, x16, x16
// CHECK-FIX:     236004:        14003826        b       0x24409c
// CHECK-NOFIX:   236004:        f940060e        ldr     x14, [x16, #8]
// CHECK-NEXT:    236008:        d65f03c0        ret
        .section .text.19, "ax", %progbits
        .balign 4096
        .globl t4_ff8_stppre
        .type t4_ff8_stppre, %function
        .space 4096 - 8
t4_ff8_stppre:
        adrp x16, dat1
        stp x1,x2, [x3, #16]!
        mul x3, x16, x16
        ldr x14, [x16, #8]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 237FF8 in unpatched output.
// CHECK: <t4_ff8_stppost>:
// CHECK-NEXT:    237ff8:        b0000150        adrp    x16, 0x260000
// CHECK-NEXT:    237ffc:        a8810861        stp     x1, x2, [x3], #16
// CHECK-NEXT:    238000:        9b107e03        mul             x3, x16, x16
// CHECK-FIX:     238004:        14003028        b       0x2440a4
// CHECK-NOFIX:   238004:        f940060e        ldr     x14, [x16, #8]
// CHECK-NEXT:    238008:        d65f03c0        ret
        .section .text.20, "ax", %progbits
        .balign 4096
        .globl t4_ff8_stppost
        .type t4_ff8_stppost, %function
        .space 4096 - 8
t4_ff8_stppost:
        adrp x16, dat2
        stp x1,x2, [x3], #16
        mul x3, x16, x16
        ldr x14, [x16, #8]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 239FFC in unpatched output.
// CHECK: <t4_ffc_stpsimd>:
// CHECK-NEXT:    239ffc:        f0000130        adrp    x16, 0x260000
// CHECK-NEXT:    23a000:        ad000861        stp             q1, q2, [x3]
// CHECK-NEXT:    23a004:        9b107e03        mul             x3, x16, x16
// CHECK-FIX:     23a008:        14002829        b       0x2440ac
// CHECK-NOFIX:   23a008:        f940060e        ldr     x14, [x16, #8]
// CHECK-NEXT:    23a00c:        d65f03c0        ret
        .section .text.21, "ax", %progbits
        .balign 4096
        .globl t4_ffc_stpsimd
        .type t4_ffc_stpsimd, %function
        .space 4096 - 4
t4_ffc_stpsimd:
        adrp x16, dat3
        stp q1,q2, [x3, #0]
        mul x3, x16, x16
        ldr x14, [x16, #8]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 23BFFC in unpatched output.
// CHECK: <t4_ffc_stnp>:
// CHECK-NEXT:    23bffc:        b0000127        adrp    x7, 0x260000
// CHECK-NEXT:    23c000:        a8000861        stnp            x1, x2, [x3]
// CHECK-NEXT:    23c004:        d503201f        nop
// CHECK-FIX:     23c008:        1400202b        b       0x2440b4
// CHECK-NOFIX:   23c008:        f94000ea        ldr             x10, [x7]
// CHECK-NEXT:    23c00c:        d65f03c0        ret
        .section .text.22, "ax", %progbits
        .balign 4096
        .globl t4_ffc_stnp
        .type t4_ffc_stnp, %function
        .space 4096 - 4
t4_ffc_stnp:
        adrp x7, dat1
        stnp x1,x2, [x3, #0]
        nop
        ldr x10, [x7, :got_lo12:dat1]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 23DFFC in unpatched output.
// CHECK: <t4_ffc_st1>:
// CHECK-NEXT:    23dffc:        f0000118        adrp    x24, 0x260000
// CHECK-NEXT:    23e000:        4d008020        st1 { v0.s }[2], [x1]
// CHECK-NEXT:    23e004:        f94006f6        ldr     x22, [x23, #8]
// CHECK-FIX:     23e008:        1400182d        b       0x2440bc
// CHECK-NOFIX:   23e008:        f93fff18        str     x24, [x24, #32760]
// CHECK-NEXT:    23e00c:        d65f03c0        ret
        .section .text.23, "ax", %progbits
        .balign 4096
        .globl t4_ffc_st1
        .type t4_ffc_st1, %function
        .space 4096 - 4
t4_ffc_st1:
        adrp x24, dat2
        st1 { v0.s }[2], [x1]
        ldr x22, [x23, :got_lo12:dat2]
        str x24, [x24, #32760]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 23FFF8 in unpatched output.
// CHECK: <t3_ff8_ldr_once>:
// CHECK-NEXT:    23fff8:        b0000100        adrp    x0, 0x260000
// CHECK-NEXT:    23fffc:        4c827020        st1     { v0.16b }, [x1], x2
// CHECK-FIX:     240000:        14001031        b       0x2440c4
// CHECK-NOFIX:   240000:        f9400801        ldr     x1, [x0, #16]
// CHECK-NEXT:    240004:        f9400802        ldr     x2, [x0, #16]
// CHECK-NEXT:    240008:        d65f03c0        ret
        .section .text.24, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldr_once
        .type t3_ff8_ldr_once, %function
        .space 4096 - 8
t3_ff8_ldr_once:
        adrp x0, dat3
        st1 { v0.16b }, [x1], x2
        ldr x1, [x0, #16]
        ldr x2, [x0, #16]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 241FF8 in unpatched output.
// CHECK: <t3_ff8_ldxr>:
// CHECK-NEXT:    241ff8:        f00000e0        adrp    x0, 0x260000
// CHECK-NEXT:    241ffc:        c85f7c03        ldxr    x3, [x0]
// CHECK-FIX:     242000:        14000833        b       0x2440cc
// CHECK-NOFIX:   242000:        f9400801        ldr     x1, [x0, #16]
// CHECK:         242004:        f9400802        ldr     x2, [x0, #16]
// CHECK-NEXT:    242008:        d65f03c0        ret
        .section .text.25, "ax", %progbits
        .balign 4096
        .globl t3_ff8_ldxr
        .type t3_ff8_ldxr, %function
        .space 4096 - 8
t3_ff8_ldxr:
        adrp x0, dat3
        ldxr x3, [x0]
        ldr x1, [x0, #16]
        ldr x2, [x0, #16]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 243FF8 in unpatched output.
// CHECK: <t3_ff8_stxr>:
// CHECK-NEXT:    243ff8:        b00000e0        adrp    x0, 0x260000
// CHECK-NEXT:    243ffc:        c8047c03        stxr    w4, x3, [x0]
// CHECK-FIX:     244000:        14000035        b       0x2440d4
// CHECK-NOFIX:   244000:        f9400801        ldr     x1, [x0, #16]
// CHECK:         244004:        f9400802        ldr     x2, [x0, #16]
// CHECK-NEXT:    244008:        d65f03c0        ret
        .section .text.26, "ax", %progbits
        .balign 4096
        .globl t3_ff8_stxr
        .type t3_ff8_stxr, %function
        .space 4096 - 8
t3_ff8_stxr:
        adrp x0, dat3
        stxr w4, x3, [x0]
        ldr x1, [x0, #16]
        ldr x2, [x0, #16]
        ret

        .text
        .globl _start
        .type _start, %function
_start:
        ret

// CHECK-FIX: <__CortexA53843419_212000>:
// CHECK-FIX-NEXT:    24400c:    f9400000        ldr     x0, [x0]
// CHECK-FIX-NEXT:    244010:    17ff37fd        b       0x212004
// CHECK-FIX: <__CortexA53843419_214000>:
// CHECK-FIX-NEXT:    244014:    f9400402        ldr     x2, [x0, #8]
// CHECK-FIX-NEXT:    244018:    17ff3ffb        b       0x214004
// CHECK-FIX: <__CortexA53843419_216004>:
// CHECK-FIX-NEXT:    24401c:    f9400803        ldr     x3, [x0, #16]
// CHECK-FIX-NEXT:    244020:    17ff47fa        b       0x216008
// CHECK-FIX: <__CortexA53843419_218000>:
// CHECK-FIX-NEXT:    244024:    f9400c02        ldr     x2, [x0, #24]
// CHECK-FIX-NEXT:    244028:    17ff4ff7        b       0x218004
// CHECK-FIX: <__CortexA53843419_21A004>:
// CHECK-FIX-NEXT:    24402c:    f900139c        str     x28, [x28, #32]
// CHECK-FIX-NEXT:    244030:    17ff57f6        b       0x21a008
// CHECK-FIX: <__CortexA53843419_21C004>:
// CHECK-FIX-NEXT:    244034:    f9001784        str     x4, [x28, #40]
// CHECK-FIX-NEXT:    244038:    17ff5ff4        b       0x21c008
// CHECK-FIX: <__CortexA53843419_21E000>:
// CHECK-FIX-NEXT:    24403c:    f94003bd        ldr     x29, [x29]
// CHECK-FIX-NEXT:    244040:    17ff67f1        b       0x21e004
// CHECK-FIX: <__CortexA53843419_220004>:
// CHECK-FIX-NEXT:    244044:    f94007bd        ldr     x29, [x29, #8]
// CHECK-FIX-NEXT:    244048:    17ff6ff0        b       0x220008
// CHECK-FIX: <__CortexA53843419_222004>:
// CHECK-FIX-NEXT:    24404c:    f9400a41        ldr     x1, [x18, #16]
// CHECK-FIX-NEXT:    244050:    17ff77ee        b       0x222008
// CHECK-FIX: <__CortexA53843419_224000>:
// CHECK-FIX-NEXT:    244054:    f9400e52        ldr     x18, [x18, #24]
// CHECK-FIX-NEXT:    244058:    17ff7feb        b       0x224004
// CHECK-FIX: <__CortexA53843419_226004>:
// CHECK-FIX-NEXT:    24405c:    f94011ea        ldr     x10, [x15, #32]
// CHECK-FIX-NEXT:    244060:    17ff87ea        b       0x226008
// CHECK-FIX: <__CortexA53843419_228000>:
// CHECK-FIX-NEXT:    244064:    f940160d        ldr     x13, [x16, #40]
// CHECK-FIX-NEXT:    244068:    17ff8fe7        b       0x228004
// CHECK-FIX: <__CortexA53843419_22A004>:
// CHECK-FIX-NEXT:    24406c:    f9400ce9        ldr     x9, [x7, #24]
// CHECK-FIX-NEXT:    244070:    17ff97e6        b       0x22a008
// CHECK-FIX: <__CortexA53843419_22C004>:
// CHECK-FIX-NEXT:    244074:    f94012f6        ldr     x22, [x23, #32]
// CHECK-FIX-NEXT:    244078:    17ff9fe4        b       0x22c008
// CHECK-FIX: <__CortexA53843419_22E000>:
// CHECK-FIX-NEXT:    24407c:    f94016f8        ldr     x24, [x23, #40]
// CHECK-FIX-NEXT:    244080:    17ffa7e1        b       0x22e004
// CHECK-FIX: <__CortexA53843419_230004>:
// CHECK-FIX-NEXT:    244084:    f9400002        ldr     x2, [x0]
// CHECK-FIX-NEXT:    244088:    17ffafe0        b       0x230008
// CHECK-FIX: <__CortexA53843419_232008>:
// CHECK-FIX-NEXT:    24408c:    f900079b        str     x27, [x28, #8]
// CHECK-FIX-NEXT:    244090:    17ffb7df        b       0x23200c
// CHECK-FIX: <__CortexA53843419_234004>:
// CHECK-FIX-NEXT:    244094:    f9400a0e        ldr     x14, [x16, #16]
// CHECK-FIX-NEXT:    244098:    17ffbfdc        b       0x234008
// CHECK-FIX: <__CortexA53843419_236004>:
// CHECK-FIX-NEXT:    24409c:    f940060e        ldr     x14, [x16, #8]
// CHECK-FIX-NEXT:    2440a0:    17ffc7da        b       0x236008
// CHECK-FIX: <__CortexA53843419_238004>:
// CHECK-FIX-NEXT:    2440a4:    f940060e        ldr     x14, [x16, #8]
// CHECK-FIX-NEXT:    2440a8:    17ffcfd8        b       0x238008
// CHECK-FIX: <__CortexA53843419_23A008>:
// CHECK-FIX-NEXT:    2440ac:    f940060e        ldr     x14, [x16, #8]
// CHECK-FIX-NEXT:    2440b0:    17ffd7d7        b       0x23a00c
// CHECK-FIX: <__CortexA53843419_23C008>:
// CHECK-FIX-NEXT:    2440b4:    f94000ea        ldr     x10, [x7]
// CHECK-FIX-NEXT:    2440b8:    17ffdfd5        b       0x23c00c
// CHECK-FIX: <__CortexA53843419_23E008>:
// CHECK-FIX-NEXT:    2440bc:    f93fff18        str     x24, [x24, #32760]
// CHECK-FIX-NEXT:    2440c0:    17ffe7d3        b       0x23e00c
// CHECK-FIX: <__CortexA53843419_240000>:
// CHECK-FIX-NEXT:    2440c4:    f9400801        ldr     x1, [x0, #16]
// CHECK-FIX-NEXT:    2440c8:    17ffefcf        b       0x240004
// CHECK-FIX: <__CortexA53843419_242000>:
// CHECK-FIX-NEXT:    2440cc:    f9400801        ldr     x1, [x0, #16]
// CHECK-FIX-NEXT:    2440d0:    17fff7cd        b       0x242004
// CHECK-FIX: <__CortexA53843419_244000>:
// CHECK-FIX-NEXT:    2440d4:    f9400801        ldr     x1, [x0, #16]
// CHECK-FIX-NEXT:    2440d8:    17ffffcb        b       0x244004
        .data
        .globl dat1
        .globl dat2
        .globl dat3
dat1:   .quad 1
dat2:   .quad 2
dat3:   .quad 3
