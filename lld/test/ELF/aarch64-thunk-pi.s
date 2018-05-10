// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       .text_low : { *(.text_low) } \
// RUN:       .text_high 0x10000000 : { *(.text_high) } \
// RUN:       } " > %t.script
// RUN: ld.lld --script %t.script --shared %t -o %t2 2>&1
// RUN: llvm-objdump -d -triple=aarch64-linux-gnu %t2 | FileCheck %s
// REQUIRES: aarch64

// Check that Position Independent thunks are generated for shared libraries.
 .section .text_low, "ax", %progbits
 .globl low_target
 .type low_target, %function
low_target:
 // Need thunk to high_target@plt
 bl high_target
 ret
// CHECK: low_target:
// CHECK-NEXT:        0:        04 00 00 94     bl      #16
// CHECK-NEXT:        4:        c0 03 5f d6     ret

 .hidden low_target2
 .globl low_target2
 .type low_target2, %function
low_target2:
 // Need thunk to high_target
 bl high_target2
 ret
// CHECK: low_target2:
// CHECK-NEXT:        8:        05 00 00 94     bl      #20
// CHECK-NEXT:        c:        c0 03 5f d6     ret

// Expect range extension thunks for .text_low
// adrp calculation is (PC + signed immediate) & (!0xfff)
// CHECK: __AArch64ADRPThunk_high_target:
// CHECK-NEXT:       70:       10 00 08 90     adrp    x16, #268435456
// CHECK-NEXT:       74:       10 02 03 91     add     x16, x16, #192
// CHECK-NEXT:       78:       00 02 1f d6     br      x16
// CHECK: __AArch64ADRPThunk_high_target2:
// CHECK-NEXT:       7c:       10 00 08 90     adrp    x16, #268435456
// CHECK-NEXT:       80:       10 22 00 91     add     x16, x16, #8
// CHECK-NEXT:       84:       00 02 1f d6     br      x16


 .section .text_high, "ax", %progbits
 .globl high_target
 .type high_target, %function
high_target:
 // No thunk needed as we can reach low_target@plt
 bl low_target
 ret
// CHECK: high_target:
// CHECK-NEXT: 10000000:        34 00 00 94     bl      #208
// CHECK-NEXT: 10000004:        c0 03 5f d6     ret

 .hidden high_target2
 .globl high_target2
 .type high_target2, %function
high_target2:
 // Need thunk to low_target
 bl low_target2
 ret
// CHECK: high_target2:
// CHECK-NEXT: 10000008:        02 00 00 94     bl      #8
// CHECK-NEXT: 1000000c:        c0 03 5f d6     ret

// Expect Thunk for .text.high

// CHECK: __AArch64ADRPThunk_low_target2:
// CHECK-NEXT: 10000010:	10 00 f8 90 	adrp	x16, #-268435456
// CHECK-NEXT: 10000014:	10 a2 01 91 	add	x16, x16, #104
// CHECK-NEXT: 10000018:	00 02 1f d6 	br	x16

// CHECK: Disassembly of section .plt:
// CHECK-NEXT: .plt:
// CHECK-NEXT: 100000a0:       f0 7b bf a9     stp     x16, x30, [sp, #-16]!
// CHECK-NEXT: 100000a4:       10 00 00 90     adrp    x16, #0
// CHECK-NEXT: 100000a8:       11 7a 40 f9     ldr     x17, [x16, #240]
// CHECK-NEXT: 100000ac:       10 c2 03 91     add     x16, x16, #240
// CHECK-NEXT: 100000b0:       20 02 1f d6     br      x17
// CHECK-NEXT: 100000b4:       1f 20 03 d5     nop
// CHECK-NEXT: 100000b8:       1f 20 03 d5     nop
// CHECK-NEXT: 100000bc:       1f 20 03 d5     nop
// CHECK-NEXT: 100000c0:       10 00 00 90     adrp    x16, #0
// CHECK-NEXT: 100000c4:       11 7e 40 f9     ldr     x17, [x16, #248]
// CHECK-NEXT: 100000c8:       10 e2 03 91     add     x16, x16, #248
// CHECK-NEXT: 100000cc:       20 02 1f d6     br      x17
// CHECK-NEXT: 100000d0:       10 00 00 90     adrp    x16, #0
// CHECK-NEXT: 100000d4:       11 82 40 f9     ldr     x17, [x16, #256]
// CHECK-NEXT: 100000d8:       10 02 04 91     add     x16, x16, #256
// CHECK-NEXT: 100000dc:       20 02 1f d6     br      x17
