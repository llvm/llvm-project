// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2 2>&1
// RUN: llvm-readobj -sections %t2 | FileCheck %s
// REQUIRES: arm

// Check that only a single .ARM.exidx output section is created when
// there are input sections of the form .ARM.exidx.<section-name>. The
// assembler creates the .ARM.exidx input sections with the .cantunwind
// directive
 .syntax unified
 .section .text, "ax",%progbits
 .globl _start
 .align 2
 .type _start,%function
_start:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.f1, "ax", %progbits
 .globl f1
 .align 2
 .type f1,%function
f1:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.f2, "ax", %progbits
 .globl f2
 .align 2
 .type f2,%function
f2:
 .fnstart
 bx lr
 .cantunwind
 .fnend

// CHECK:         Section {
// CHECK:         Name: .ARM.exidx
// CHECK-NEXT:    Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-NEXT:    Flags [ (0x82)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:      SHF_LINK_ORDER (0x80)
// CHECK-NEXT:    ]

// CHECK-NOT:     Name: .ARM.exidx.text.f1
// CHECK-NOT:     Name: .ARM.exidx.text.f2
