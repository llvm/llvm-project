## Check that unclaimed PC-relative relocation from data to code is detected
## and reported to the user.

# REQUIRES: system-linux

# RUN: %clang %cflags -no-pie %s -o %t.exe -Wl,-q -nostartfiles
# RUN: not llvm-bolt %t.exe -o %t.bolt --strict 2>&1 | FileCheck %s

# CHECK: BOLT-ERROR: 1 unclaimed PC-relative relocation(s) left in data

  .text
  .globl _start
  .type _start, %function
_start:
  movl $42, %eax
.L0:
  ret
  .size _start, .-_start

## Force relocation mode.
  .reloc 0, R_X86_64_NONE

  .section .rodata
  .long .L0-.
