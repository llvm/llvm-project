/// NOTE: assign section addresses explicitly to make the symbol difference
/// calculation below less fragile.
// RUN: %clang %cflags -Wl,--image-base=0,--section-start=.text=0x1000,--section-start=.data=0x2000 -o %t %s
// RUN: llvm-bolt -o %t.bolt %t
// RUN: llvm-readelf -x .data %t.bolt | FileCheck %s

  .text

  .globl _start
  .p2align 1
_start:
.LBB0_0:
  auipc a1, %pcrel_hi(.LJTI0_0)
  addi a1, a1, %pcrel_lo(.LBB0_0)
  lw a0, (a1)
  add a0, a0, a1
  jr a0
.LBB0_1:
  ret
  .size _start, .-_start

  .data
/// .LJTI0_0 = 0x2000
/// .LBB0_1 = 0x40000e
// CHECK: Hex dump of section '.data':
// CHECK-NEXT: 0x00002000 0ee03f00
.LJTI0_0:
  .word .LBB0_1 - .LJTI0_0
