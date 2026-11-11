// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7-windows %s -o %t.obj
// RUN: lld-link %t.obj %p/Inputs/library.lib /subsystem:console /entry:main \
// RUN:   /delayload:library.dll /out:%t.exe
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x150101a \
// RUN:   --stop-address=0x150104a | FileCheck %s

// __delayLoadHelper2 is more than 16 MiB before the tail merge, out of reach of
// a bl. The bl goes to a range extension thunk right after the tail merge.

// CHECK:      150101a: {{.*}} push.w {r0, r1, r2, r3, r11, lr}
// CHECK:      1501030: f000 f806 bl 0x1501040
// CHECK:      1501040: f64f 7cc0 movw r12, #65472
// CHECK-NEXT: 1501044: f6cf 6cef movt r12, #65263
// CHECK-NEXT: 1501048: 44e7 add pc, r12

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
