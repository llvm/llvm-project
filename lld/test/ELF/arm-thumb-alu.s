// REQUIRES: arm

// RUN: split-file %s %t
// RUN: llvm-mc -filetype=obj -triple=thumbv6m-unknown-linux-gnueabi %t/asm -o %t.o
// RUN: ld.lld --script %t/lds %t.o -o %t2
// RUN: llvm-objdump -d %t2 --triple=thumbv6m-unknown-linux-gnueabi --no-show-raw-insn | FileCheck %s

//--- lds
SECTIONS {
  .tests 0x00001000 : AT(0x00001000) { *(.tests) }
  .sym1  0x11223344 : AT(0x11223344) { *(.sym1) }
  .sym2  0x00ffffff : AT(0x00ffffff) { *(.sym2) }
  .fn    0x55667788 : AT(0x55667788) { *(.fn) }
}

//--- asm
  .section .tests, "ax", %progbits

// CHECK-LABEL: <R_ARM_THM_ALU_ABS_G0_NC>:
// CHECK:      adds    r0, #0x44
// CHECK-NEXT: movs    r0, #0x44
// CHECK-NEXT: movs    r0, #0x45
// CHECK-NEXT: movs    r0, #0x43
// CHECK-NEXT: movs    r0, #0xff
// CHECK-NEXT: movs    r0, #0x0
// CHECK-NEXT: movs    r0, #0xfe
// CHECK-NEXT: movs    r0, #0x89
// CHECK-NEXT: movs    r0, #0x8b
R_ARM_THM_ALU_ABS_G0_NC:
  adds r0, :lower0_7:sym1
  movs r0, :lower0_7:sym1
  movs r0, :lower0_7:sym1+1
  movs r0, :lower0_7:sym1+0xff
  movs r0, :lower0_7:sym2
  movs r0, :lower0_7:sym2+1
  movs r0, :lower0_7:sym2+0xff
  movs r0, :lower0_7:fn
  movs r0, :lower0_7:fn+2

// CHECK-LABEL: <R_ARM_THM_ALU_ABS_G1_NC>:
// CHECK:      adds    r0, #0x33
// CHECK-NEXT: movs    r0, #0x33
// CHECK-NEXT: movs    r0, #0x33
// CHECK-NEXT: movs    r0, #0x34
// CHECK-NEXT: movs    r0, #0xff
// CHECK-NEXT: movs    r0, #0x0
// CHECK-NEXT: movs    r0, #0x0
// CHECK-NEXT: movs    r0, #0x77
// CHECK-NEXT: movs    r0, #0x77
R_ARM_THM_ALU_ABS_G1_NC:
  adds r0, :lower8_15:sym1
  movs r0, :lower8_15:sym1
  movs r0, :lower8_15:sym1+1
  movs r0, :lower8_15:sym1+0xff
  movs r0, :lower8_15:sym2
  movs r0, :lower8_15:sym2+1
  movs r0, :lower8_15:sym2+0xff
  movs r0, :lower8_15:fn
  movs r0, :lower8_15:fn+2

// CHECK-LABEL: <R_ARM_THM_ALU_ABS_G2_NC>:
// CHECK:      adds    r0, #0x22
// CHECK-NEXT: movs    r0, #0x22
// CHECK-NEXT: movs    r0, #0x22
// CHECK-NEXT: movs    r0, #0x22
// CHECK-NEXT: movs    r0, #0xff
// CHECK-NEXT: movs    r0, #0x0
// CHECK-NEXT: movs    r0, #0x0
// CHECK-NEXT: movs    r0, #0x66
// CHECK-NEXT: movs    r0, #0x66
R_ARM_THM_ALU_ABS_G2_NC:
  adds r0, :upper0_7:sym1
  movs r0, :upper0_7:sym1
  movs r0, :upper0_7:sym1+1
  movs r0, :upper0_7:sym1+0xff
  movs r0, :upper0_7:sym2
  movs r0, :upper0_7:sym2+1
  movs r0, :upper0_7:sym2+0xff
  movs r0, :upper0_7:fn
  movs r0, :upper0_7:fn+2

// CHECK-LABEL: <R_ARM_THM_ALU_ABS_G3>:
// CHECK:      adds    r0, #0x11
// CHECK-NEXT: movs    r0, #0x11
// CHECK-NEXT: movs    r0, #0x11
// CHECK-NEXT: movs    r0, #0x11
// CHECK-NEXT: movs    r0, #0x0
// CHECK-NEXT: movs    r0, #0x1
// CHECK-NEXT: movs    r0, #0x1
// CHECK-NEXT: movs    r0, #0x55
// CHECK-NEXT: movs    r0, #0x55
R_ARM_THM_ALU_ABS_G3:
  adds r0, :upper8_15:sym1
  movs r0, :upper8_15:sym1
  movs r0, :upper8_15:sym1+1
  movs r0, :upper8_15:sym1+0xff
  movs r0, :upper8_15:sym2
  movs r0, :upper8_15:sym2+1
  movs r0, :upper8_15:sym2+0xff
  movs r0, :upper8_15:fn
  movs r0, :upper8_15:fn+2

  .section .sym1, "aw", %progbits
sym1:
  .byte 0

  .section .sym2, "aw", %progbits
sym2:
  .byte 0

  .section .fn, "ax", %progbits
  .thumb_func
fn:
  bx lr
