/// FIXME llvm-mc is used instead of clang because we need a recent change in
/// the RISC-V MC layer (D153260). Once that one is released, we can switch to
/// clang. (Note that the pre-merge check buildbots use the system's clang).
// RUN: llvm-mc -triple riscv64 -mattr=+c -filetype obj -o %t.o %s
// RUN: ld.lld -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.bolt %t 2>&1 | FileCheck %s
// RUN: llvm-objdump -d %t.bolt | FileCheck --check-prefix=CHECK-OBJDUMP %s

// CHECK-NOT: BOLT-WARNING

/// Check that .word is not disassembled by BOLT
// CHECK: 00000000: nop
// CHECK: 00000002: ret

/// Check .word is still present in output
// CHECK-OBJDUMP: <_start>:
// CHECK-OBJDUMP-NEXT: nop
// CHECK-OBJDUMP-NEXT: unimp
// CHECK-OBJDUMP-NEXT: unimp
// CHECK-OBJDUMP-NEXT: ret
    .text
    .globl _start
    .p2align 1
_start:
    nop
    .word 0x0
    ret
