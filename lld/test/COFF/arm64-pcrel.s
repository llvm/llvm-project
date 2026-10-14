// REQUIRES: aarch64
// RUN: llvm-mc -triple aarch64-unknown-windows-msvc -filetype obj %s -o %t.o
// RUN: lld-link -machine:arm64 -dll -noentry -base:0x0  %t.o -out:%t.dll
// RUN: llvm-objdump -d %t.dll | FileCheck %s

// CHECK: 	0000000000001000 <.text>:
// CHECK-NEXT:  1000: d503201f      nop
// CHECK-NEXT:  1004: 14000002      b       0x100c <.text+0xc>
// CHECK-NEXT:  1008: d503201f      nop
// CHECK-NEXT:  100c: d503201f      nop
// CHECK-NEXT:  1010: d65f03c0      ret

.globl main

main:
    nop
    b .Lpcrel_target-4

    .def .Lpcrel_target
    .scl 3
    .type 32
    .p2align 2
    .endef
    nop
    nop
.Lpcrel_target:
    ret
