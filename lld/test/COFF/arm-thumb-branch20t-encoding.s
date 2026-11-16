// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7-windows %s -o %t.obj
// RUN: lld-link -entry:start -subsystem:console %t.obj -out:%t.exe
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x401000 --stop-address=0x401010 | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x4e2002 --stop-address=0x4e200a | FileCheck --check-prefix=BACK %s

// The branches below cover all four values of offset bits 19 and 18 (J2 and
// J1), forward and backward. With J1 and J2 swapped, some land 256 KB away.

// CHECK:      401000: {{.*}}bne.w 0x41a000
// CHECK-NEXT: 401004: {{.*}}bne.w 0x44c000
// CHECK-NEXT: 401008: {{.*}}bne.w 0x497000
// CHECK-NEXT: 40100c: {{.*}}bne.w 0x4e2000

// BACK:      4e2002: {{.*}}bne.w 0x497000
// BACK-NEXT: 4e2006: {{.*}}bne.w 0x44c000

    .syntax unified
    .text
    .def start; .scl 2; .type 32; .endef
    .globl start
start:
    bne.w t100
    bne.w t300
    bne.w t600
    bne.w t900
    bx lr
    .section .text$a,"xr"
    .space 100*1024 - 18
    .section .text$b,"xr"
    .def t100; .scl 2; .type 32; .endef
    .globl t100
t100:
    bx lr
    .space 200*1024 - 2
    .def t300; .scl 2; .type 32; .endef
    .globl t300
t300:
    bx lr
    .space 300*1024 - 2
    .def t600; .scl 2; .type 32; .endef
    .globl t600
t600:
    bx lr
    .space 300*1024 - 2
    .def t900; .scl 2; .type 32; .endef
    .globl t900
t900:
    bx lr
    bne.w t600
    bne.w t300
    bx lr
