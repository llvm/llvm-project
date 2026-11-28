// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7-windows %s -o %t.obj
// RUN: lld-link %t.obj %p/Inputs/library.lib /subsystem:console /entry:main \
// RUN:   /delayload:library.dll /out:%t.exe
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x150101a \
// RUN:   --stop-address=0x1501046 | FileCheck %s

// __delayLoadHelper2 is more than 16 MiB before the tail merge, out of reach of
// a bl. The tail merge loads its address, with the Thumb bit set, and uses blx.

// CHECK:      150101a: {{.*}} push.w {r0, r1, r2, r3, r11, lr}
// CHECK:      1501030: f241 020d movw r2, #4109
// CHECK-NEXT: 1501034: f2c0 0240 movt r2, #64
// CHECK-NEXT: 1501038: 4790 blx r2

    .syntax unified
    .text
    .globl main
    .def main; .scl 2; .type 32; .endef
main:
    movw r0, :lower16:__imp_function
    movt r0, :upper16:__imp_function
    ldr r0, [r0]
    bx r0

    .globl __delayLoadHelper2
    .def __delayLoadHelper2; .scl 2; .type 32; .endef
__delayLoadHelper2:
    bx lr

    .section .text$z, "xr"
    .space 0x1100000
