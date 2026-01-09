## Check that llvm-bolt avoids optimization of functions referenced from
## __rseq_cs section, i.e. containing critical sections and abort handlers used
## by restartable sequences in tcmalloc.

# RUN: %clang %cflags %s -o %t -nostdlib -no-pie -Wl,-q
# RUN: llvm-bolt %t -o %t.bolt --print-cfg 2>&1 | FileCheck %s
# RUN: %clang %cflags %s -o %t.pie -nostdlib -pie -Wl,-q
# RUN: llvm-bolt %t.pie -o %t.pie.bolt 2>&1 | FileCheck %s

# CHECK: restartable sequence reference detected in _start
# CHECK: restartable sequence reference detected in __rseq_abort

  .text
.reloc 0, R_AARCH64_NONE

  .global _start
  .type _start, %function
_start:
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp
.L1:
  ldp     x29, x30, [sp], #16
.L2:
  ret
  .size _start, .-_start

  .section __rseq_abort, "ax"
## Signature for rseq abort IP. Unmarked in the symbol table.
 .long 0xd428bc00 // RSEQ_SIG_CODE, BRK #0x45E0
.L3:
  b .L2

.section __rseq_cs, "aw"
.balign 32
  .quad .L1
  .quad .L3
