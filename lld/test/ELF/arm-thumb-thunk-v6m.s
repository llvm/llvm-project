// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv6m-none-eabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       . = SIZEOF_HEADERS; \
// RUN:       .text_low : { *(.text_low) *(.text_low2) } \
// RUN:       .text_high 0x2000000 : { *(.text_high) *(.text_high2) } \
// RUN:       } " > %t.script
// RUN: ld.lld --no-rosegment --script %t.script %t -o %t2
// RUN: llvm-objdump -d %t2 --triple=armv6m-none-eabi | FileCheck %s
// RUN: ld.lld --no-rosegment --script %t.script %t -o %t3 --pie
// RUN: llvm-objdump -d %t3 --triple=armv6m-none-eabi | FileCheck --check-prefix=CHECK-PI %s

// Range extension thunks for Arm Architecture v6m. Only Thumb instructions
// are permitted which limits the access to instructions that can access the
// high registers (r8 - r15), this means that the thunks have to spill
// low registers (r0 - r7) in order to perform the transfer of control.

 .syntax unified
 .section .text_low, "ax", %progbits
 .thumb
 .type _start, %function
 .balign 4
 .globl _start
_start:
 bl far

 .section .text_high, "ax", %progbits
 .globl far
 .type far, %function
far:
 bx lr

// CHECK: Disassembly of section .text_low:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:       94:        f000 f800       bl      0x98 <__Thumbv6MABSLongThunk_far>
// CHECK: <__Thumbv6MABSLongThunk_far>:
// CHECK-NEXT:       98:        b403    push    {r0, r1}
// CHECK-NEXT:       9a:        4801    ldr     r0, [pc, #4]
// CHECK-NEXT:       9c:        9001    str     r0, [sp, #4]
// CHECK-NEXT:       9e:        bd01    pop     {r0, pc}
// CHECK:       a0:     01 00 00 02     .word   0x02000001
// CHECK: Disassembly of section .text_high:
// CHECK-EMPTY:
// CHECK-NEXT: <far>:
// CHECK-NEXT:  2000000:        4770    bx      lr

// CHECK-PI: Disassembly of section .text_low:
// CHECK-PI-EMPTY:
// CHECK-PI-NEXT: <_start>:
// CHECK-PI-NEXT:      130:     f000 f800       bl      0x134 <__Thumbv6MPILongThunk_far>
// CHECK-PI: <__Thumbv6MPILongThunk_far>:
// CHECK-PI-NEXT:      134:     b401    push    {r0}
// CHECK-PI-NEXT:      136:     4802    ldr     r0, [pc, #8]
// CHECK-PI-NEXT:      138:     4684    mov     r12, r0
// CHECK-PI-NEXT:      13a:     bc01    pop     {r0}
// pc = pc (0x13c + 4) + r12 (1fffec1) = 0x2000001 = .far
// CHECK-PI-NEXT:      13c:     44e7    add     pc, r12
// CHECK-PI-NEXT:      13e:     46c0    mov     r8, r8
// CHECK-PI:           140:     c1 fe ff 01     .word   0x01fffec1

// CHECK-PI: Disassembly of section .text_high:
// CHECK-PI-EMPTY:
// CHECK-PI-NEXT: <far>:
// CHECK-PI-NEXT:  2000000:     4770    bx      lr
