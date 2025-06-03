## Check that llvm-bolt generates a proper error message when ADR instruction
## cannot be relaxed.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: not llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s

# CHECK: BOLT-ERROR: cannot relax ADR in non-simple function _start

## The function contains unknown control flow and llvm-bolt fails to recover
## CFG. As BOLT has to preserve the function layout, the ADR instruction cannot
## be relaxed into ADRP+ADD.
  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  adr x1, foo
  br x0
  ret  x0
  .cfi_endproc
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  cmp  x1, x11
  b.hi  .L2
  mov  x0, #0x0
.L2:
  adr x1, foo
  ret  x30
  .cfi_endproc
  .size foo, .-foo
