## Verify that llvm-bolt removes nop instructions from functions with indirect
## branches that have defined control flow.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --print-normalized 2>&1 | FileCheck %s
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt \
# RUN:   | FileCheck %s --check-prefix=CHECK-OBJDUMP

# CHECK-OBJDUMP-LABEL: _start
# CHECK-OBJDUMP-NOT: nop

  .section .text
  .align 4
  .globl _start
  .type  _start, %function
_start:
# CHECK-LABEL: Binary Function "_start"
  nop
# CHECK-NOT: nop
  br      x0
# CHECK: br x0 # TAILCALL
.size _start, .-_start

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
