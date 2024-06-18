// Test how BOLT handles indirect branch sequence of instructions in
// AArch64MCPlus builder.
// This test checks that case when we have no shift amount after add
// instruction. This pattern comes from libc, so needs to build '-static'
// binary to reproduce the issue easily.
//
//   adr     x6, 0x219fb0 <sigall_set+0x88>
//   add     x6, x6, x14, lsl #2
//   ldr     w7, [x6]
//   add     x6, x6, w7, sxtw => no shift amount
//   br      x6
// It also tests another case when we use '-fuse-ld=lld' along with '-static'
// which produces the following sequence of intsructions:
//
//  nop   => nop/adr instead of adrp/add
//  adr     x13, 0x215a18 <_nl_value_type_LC_COLLATE+0x50>
//  ldrh    w13, [x13, w12, uxtw #1]
//  adr     x12, 0x247b30 <__gettextparse+0x5b0>
//  add     x13, x12, w13, sxth #2
//  br      x13

// clang-format off

// REQUIRES: system-linux
// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
// RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg \
// RUN:  -v=1 2>&1 | FileCheck %s

// CHECK: BOLT-WARNING: Failed to match indirect branch: nop/adr instead of adrp/add
// CHECK: BOLT-WARNING: Failed to match indirect branch: ShiftVAL != 2


  .section .text
  .align 4
  .globl _start
  .type  _start, %function
_start:
  bl bar
  bl end
  mov x0, #4
  mov w8, #93
  svc #0

bar:
  mov     w1, #3
  cmp     x1, #0
  b.eq    end
  nop
  adr     x3, jump_table
  ldrh    w3, [x3, x1, lsl #1]
  adr     x1, .case0
  add     x3, x1, w3, sxth #2
  br      x3
.case0:
  mov     w0, #1
  ret
.case1:
  mov     w0, #2
  ret
.case3:
  mov     w0, #3
  ret
.case4:
  nop
  mov     x1, #0
  adr     x3, datatable
  add     x3, x3, x1, lsl #2
  ldr     w2, [x3]
  add     x3, x3, w2, sxtw
  br      x3
  nop
  mov     w0, #4
  ret
.case7:
  mov     w0, #4
  ret

foo1:
   ret

foo2:
   add    w0, w0, #3
   ret

foo3:
   add    w0, w0, #3
   ret

end:
  add     x0, x0, #99
  ret

  .section .rodata,"a",@progbits
jump_table:
  .hword  (.case0-.case0)>>2
  .hword  (.case1-.case0)>>2
  .hword  (.case3-.case0)>>2
  .hword  (.case4-.case0)>>2
  .hword  (.case7-.case0)>>2


datatable:
  .word foo1-datatable
  .word foo2-datatable
  .word foo3-datatable
  .word 20
