# Checking that .cfi-negate_ra_state directives are emitted in the same location as in the input in the case of no optimizations.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags  %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.exe.bolt --no-threads --print-all | FileCheck %s --check-prefix=CHECK-BOLT

# Check that the negate-ra-state at the start of bar is not discarded.
# If it was discarded, MarkRAState would report bar as having inconsistent RAStates.
# CHECK-BOLT-NOT: BOLT-INFO: inconsistent RAStates in function foo
# CHECK-BOLT-NOT: BOLT-INFO: inconsistent RAStates in function bar

# Check that OpNegateRAState CFIs are generated correctly.
# CHECK-BOLT: Binary Function "foo" after insert-negate-ra-state-pass {
# CHECK-BOLT:         paciasp
# CHECK-BOLT-NEXT:    OpNegateRAState

# CHECK-BOLT:      DWARF CFI Instructions:
# CHECK-BOLT-NEXT:     0:  OpNegateRAState
# CHECK-BOLT-NEXT: End of Function "foo"

# CHECK-BOLT: Binary Function "bar" after insert-negate-ra-state-pass {
# CHECK-BOLT:         OpNegateRAState
# CHECK-BOLT-NEXT:    mov     x1, #0x0
# CHECK-BOLT-NEXT:    mov     x1, #0x1
# CHECK-BOLT-NEXT:    autiasp
# CHECK-BOLT-NEXT:    OpNegateRAState
# CHECK-BOLT-NEXT:    ret

# CHECK-BOLT:     DWARF CFI Instructions:
# CHECK-BOLT-NEXT:     0:  OpNegateRAState
# CHECK-BOLT-NEXT:     1:  OpNegateRAState
# CHECK-BOLT-NEXT: End of Function "bar"

# End of negate-ra-state insertion logs for foo and bar.
# CHECK: Binary Function "_start" after insert-negate-ra-state-pass {

# Check that the functions are in the new .text section
# RUN: llvm-objdump %t.exe.bolt -d -j .text | FileCheck %s --check-prefix=CHECK-OBJDUMP
# CHECK-OBJDUMP: <foo>:
# CHECK-OBJDUMP: <bar>:


  .text
  .globl  foo
  .p2align        2
  .type   foo,@function
foo:
  .cfi_startproc
  paciasp
  .cfi_negate_ra_state
  mov x1, #0
  b bar
  .cfi_endproc
  .size   foo, .-foo



  .text
  .globl  bar
  .p2align        2
  .type   bar,@function
bar:
  .cfi_startproc
  .cfi_negate_ra_state    // Indicating that RA is signed from the start of bar.
  mov x1, #0
  mov x1, #1
  autiasp
  .cfi_negate_ra_state
  ret
  .cfi_endproc
  .size   bar, .-bar

  .global _start
  .type _start, %function
_start:
  b foo
