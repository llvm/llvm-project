# Checking that after reordering BasicBlocks, the generated OpNegateRAState instructions
# are placed where the RA state is different between two consecutive instructions.
# This case demonstrates, that the input might have a different amount than the output:
# input has 4, but output only has 3.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags  %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.exe.bolt --no-threads --reorder-blocks=reverse \
# RUN: --print-cfg --print-after-lowering --print-only foo | FileCheck %s

# Check that the reordering succeeded.
# CHECK: Binary Function "foo" after building cfg {
# CHECK: BB Layout   : .LBB00, .Ltmp2, .Ltmp0, .Ltmp1
# CHECK: Binary Function "foo" after inst-lowering {
# CHECK: BB Layout   : .LBB00, .Ltmp1, .Ltmp0, .Ltmp2


# Check the generated CFIs.
# CHECK:         OpNegateRAState
# CHECK-NEXT:    mov     x2, #0x6

# CHECK:         autiasp
# CHECK-NEXT:    OpNegateRAState
# CHECK-NEXT:    ret

# CHECK:         paciasp
# CHECK-NEXT:    OpNegateRAState

# CHECK:         DWARF CFI Instructions:
# CHECK-NEXT:        0:  OpNegateRAState
# CHECK-NEXT:        1:  OpNegateRAState
# CHECK-NEXT:        2:  OpNegateRAState
# CHECK-NEXT:    End of Function "foo"

  .text
  .globl  foo
  .p2align        2
  .type   foo,@function
foo:
  .cfi_startproc
  // RA is unsigned
  mov x1, #0
  mov x1, #1
  mov x1, #2
  // jump into the signed "range"
  b .Lmiddle
.Lback:
// sign RA
  paciasp
  .cfi_negate_ra_state
  mov x2, #3
  mov x2, #4
  // skip unsigned instructions
  b .Lcont
  .cfi_negate_ra_state
.Lmiddle:
// RA is unsigned
  mov x4, #5
  b .Lback
  .cfi_negate_ra_state
.Lcont:
// continue in signed state
  mov x2, #6
  autiasp
  .cfi_negate_ra_state
  ret
  .cfi_endproc
  .size   foo, .-foo

  .global _start
  .type _start, %function
_start:
  b foo
