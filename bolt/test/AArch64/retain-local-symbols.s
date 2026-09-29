## Check that BOLT retains named local symbols (assembly labels) inside
## functions and updates their addresses to reflect the output layout.

# RUN: %clang %cflags %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt -lite=false
# RUN: llvm-nm -n %t.bolt | FileCheck %s

# CHECK: T _start
# CHECK: t loop_start
# CHECK: t loop_end
# CHECK: t helper

  .text
  .global _start
  .type _start, %function
_start:
  mov x0, #10
  bl helper
loop_start:
  sub x0, x0, #1
  cbnz x0, loop_start
loop_end:
  ret

helper:
  add x0, x0, #1
  ret

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
