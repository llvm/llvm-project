// Test how BOLT handles indirect branch sequence of instructions in
// AArch64MCPlus builder.

// clang-format off

// REQUIRES: system-linux, asserts

// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
// RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg --strict --debug-only=mcplus \
// RUN:  -v=1 2>&1 | FileCheck %s

// Pattern 1: there is no shift amount after the 'add' instruction.
//
//   adr     x6, 0x219fb0 <sigall_set+0x88>
//   add     x6, x6, x14, lsl #2
//   ldr     w7, [x6]
//   add     x6, x6, w7, sxtw => no shift amount
//   br      x6
//

// Pattern 2: nop/adr pair is used in place of adrp/add
//
//  nop   => nop/adr instead of adrp/add
//  adr     x13, 0x215a18 <_nl_value_type_LC_COLLATE+0x50>
//  ldrh    w13, [x13, w12, uxtw #1]
//  adr     x12, 0x247b30 <__gettextparse+0x5b0>
//  add     x13, x12, w13, sxth #2
//  br      x13

  .section .text
  .align 4
  .globl _start
  .type  _start, %function
_start:
  bl      test1
  bl      test2
// mov x0, #4
// mov w8, #93
// svc #0

// Pattern 1
// CHECK: BOLT-DEBUG: failed to match indirect branch: ShiftVAL != 2
  .globl test1
  .type  test1, %function
test1:
  mov     x1, #0
  adr     x3, datatable
  add     x3, x3, x1, lsl #2
  ldr     w2, [x3]
  add     x3, x3, w2, sxtw
  br      x3
test1_0:
   ret
test1_1:
   ret
test1_2:
   ret

// Pattern 2
// CHECK: BOLT-DEBUG: failed to match indirect branch: nop/adr instead of adrp/add
  .globl test2
  .type  test2, %function
test2:
  nop
  adr     x3, jump_table
  ldrh    w3, [x3, x1, lsl #1]
  adr     x1, test2_0
  add     x3, x1, w3, sxth #2
  br      x3
test2_0:
  ret
test2_1:
  ret

  .section .rodata,"a",@progbits
datatable:
  .word test1_0-datatable
  .word test1_1-datatable
  .word test1_2-datatable

jump_table:
  .hword  (test2_0-test2_0)>>2
  .hword  (test2_1-test2_0)>>2
