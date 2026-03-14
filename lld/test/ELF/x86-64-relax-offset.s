// REQUIRES: x86

/// Test we use input r_offset when deciding if R_X86_64_GOTPCRELX
/// relaxation is applicable.

// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux \
// RUN:   %p/Inputs/x86-64-relax-offset.s -o %t2.o
// RUN: ld.lld %t2.o %t.o -o %t.so -shared
// RUN: llvm-objdump --no-print-imm-hex -d %t.so | FileCheck %s

        mov foo@gotpcrel(%rip), %rax
        nop

// CHECK:      leaq    -11(%rip), %rax
// CHECK-NEXT: nop
