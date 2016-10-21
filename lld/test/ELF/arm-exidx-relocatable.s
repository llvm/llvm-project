// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/arm-exidx-cantunwind.s -o %tcantunwind
// Check that relocatable link maintains SHF_LINK_ORDER
// RUN: ld.lld -r %t %tcantunwind -o %t4 2>&1
// RUN: llvm-readobj -s %t4 | FileCheck %s
// REQUIRES: arm

// Each assembler created .ARM.exidx section has the SHF_LINK_ORDER flag set
// with the sh_link containing the section index of the executable section
// containing the function it describes. To maintain this property in
// relocatable links we pass through the .ARM.exidx section, the section it
// it has a sh_link to, and the associated relocation sections uncombined.

 .syntax unified
 .section .text, "ax",%progbits
 .globl _start
_start:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.f1, "ax", %progbits
 .globl f1
f1:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.f2, "ax", %progbits
 .globl f2
f2:
 .fnstart
 bx lr
 .cantunwind
 .fnend
 .globl f3
f3:
 .fnstart
 bx lr
 .cantunwind
 .fnend

// CHECK:         Name: .ARM.exidx
// CHECK-NEXT:    Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-NEXT:    Flags [ (0x82)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:      SHF_LINK_ORDER (0x80)
// CHECK:         Size: 24
// CHECK-NEXT:    Link: 7
// CHECK:         Name: .ARM.exidx.text.f1
// CHECK-NEXT:    Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-NEXT:    Flags [ (0x82)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:      SHF_LINK_ORDER (0x80)
// CHECK:    Size: 8
// CHECK-NEXT:    Link: 8
// CHECK:         Name: .ARM.exidx.text.f2
// CHECK-NEXT:    Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-NEXT:    Flags [ (0x82)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:      SHF_LINK_ORDER (0x80)
// CHECK:         Size: 16
// CHECK-NEXT:    Link: 9
// CHECK:         Name: .ARM.exidx.func1
// CHECK-NEXT:    Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-NEXT:    Flags [ (0x82)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:      SHF_LINK_ORDER (0x80)
// CHECK:         Size: 8
// CHECK-NEXT:    Link: 10
// CHECK:         Name: .ARM.exidx.func2
// CHECK-NEXT:    Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-NEXT:    Flags [ (0x82)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:      SHF_LINK_ORDER (0x80)
// CHECK:         Size: 8
// CHECK-NEXT:    Link: 11
// CHECK:         Name: .ARM.exidx.func3
// CHECK-NEXT:    Type: SHT_ARM_EXIDX (0x70000001)
// CHECK-NEXT:    Flags [ (0x82)
// CHECK-NEXT:      SHF_ALLOC (0x2)
// CHECK-NEXT:      SHF_LINK_ORDER (0x80)
// CHECK:         Size: 8
// CHECK-NEXT:    Link: 12
// CHECK:         Index: 7
// CHECK-NEXT:    Name: .text
// CHECK:         Index: 8
// CHECK-NEXT:    Name: .text.f1
// CHECK:         Index: 9
// CHECK-NEXT:    Name: .text.f2
// CHECK:         Index: 10
// CHECK-NEXT:    Name: .func1
// CHECK:         Index: 11
// CHECK-NEXT:    Name: .func2
// CHECK:         Index: 12
// CHECK-NEXT:    Name: .func3
