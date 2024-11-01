// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv5-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2 --shared
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --start-address=0x70000c --stop-address=0x700010 --triple=armv5-none-linux-gnueabi -d %t2 | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --start-address=0x80000c --stop-address=0x800010 -d %t2 | FileCheck %s --check-prefix=CHECK-CALL
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --start-address=0xd00020 --stop-address=0xd00060 --triple=armv5-none-linux-gnueabi -d %t2 | FileCheck %s --check-prefix=CHECK-PLT

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv5eb-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2 --shared
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --start-address=0x70000c --stop-address=0x700010 --triple=armv5eb-none-linux-gnueabi -d %t2 | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --start-address=0x80000c --stop-address=0x800010 -d %t2 | FileCheck %s --check-prefix=CHECK-CALL
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --start-address=0xd00020 --stop-address=0xd00060 --triple=armv5eb-none-linux-gnueabi -d %t2 | FileCheck %s --check-prefix=CHECK-PLT

/// When we create a thunk to a PLT entry the relocation is redirected to the
/// Thunk, changing its expression to a non-PLT equivalent. If the thunk
/// becomes unusable we need to restore the relocation expression to the PLT
/// form so that when we create a new thunk it targets the PLT.

/// Test case that checks the case:
/// - Thunk is created on pass 1 to a PLT entry for preemptible
/// - Some other Thunk added in the same pass moves the thunk to
/// preemptible out of range of its caller.
/// - New Thunk is created on pass 2 to PLT entry for preemptible

        .globl preemptible
        .globl preemptible2
.section .text.01, "ax", %progbits
.balign 0x100000
        .thumb
        .globl needsplt
        .type needsplt, %function
needsplt:
        bl preemptible
        .section .text.02, "ax", %progbits
        .space (1024 * 1024)

        .section .text.03, "ax", %progbits
        .space (1024 * 1024)

        .section .text.04, "ax", %progbits
        .space (1024 * 1024)

        .section .text.05, "ax", %progbits
        .space (1024 * 1024)

        .section .text.06, "ax", %progbits
        .space (1024 * 1024)

        .section .text.07, "ax", %progbits
        .space (1024 * 1024)
/// 0xd00040 = preemptible@plt
// CHECK:      0070000c <__ARMv4PILongBXThunk_preemptible>:
// CHECK-NEXT:   70000c: b       0xd00040

        .section .text.08, "ax", %progbits
        .space (1024 * 1024) - 4

        .section .text.10, "ax", %progbits
        .balign 2
        bl preemptible
        bl preemptible2
// CHECK-CALL: 80000c: blx     0x70000c <__ARMv4PILongBXThunk_preemptible>
        .balign 2
        .globl preemptible
        .type preemptible, %function
preemptible:
        bx lr
        .globl preemptible2
        .type preemptible2, %function
preemptible2:
        bx lr


        .section .text.11, "ax", %progbits
        .space (5 * 1024 * 1024)


// CHECK-PLT: Disassembly of section .plt:
// CHECK-PLT-EMPTY:
// CHECK-PLT-NEXT: 00d00020 <.plt>:
// CHECK-PLT-NEXT:   d00020: str     lr, [sp, #-4]!
// CHECK-PLT-NEXT:           add     lr, pc, #0, #12
// CHECK-PLT-NEXT:           add     lr, lr, #32, #20
// CHECK-PLT-NEXT:           ldr     pc, [lr, #148]!
// CHECK-PLT-NEXT:   d00030: d4 d4 d4 d4      .word   0xd4d4d4d4
// CHECK-PLT-NEXT:   d00034: d4 d4 d4 d4      .word   0xd4d4d4d4
// CHECK-PLT-NEXT:   d00038: d4 d4 d4 d4      .word   0xd4d4d4d4
// CHECK-PLT-NEXT:   d0003c: d4 d4 d4 d4      .word   0xd4d4d4d4
// CHECK-PLT-NEXT:   d00040: add     r12, pc, #0, #12
// CHECK-PLT-NEXT:   d00044: add     r12, r12, #32, #20
// CHECK-PLT-NEXT:   d00048: ldr     pc, [r12, #124]!
// CHECK-PLT-NEXT:   d0004c: d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-PLT-NEXT:   d00050: add     r12, pc, #0, #12
// CHECK-PLT-NEXT:   d00054: add     r12, r12, #32, #20
// CHECK-PLT-NEXT:   d00058: ldr     pc, [r12, #112]!
// CHECK-PLT-NEXT:   d0005c: d4 d4 d4 d4     .word   0xd4d4d4d4
