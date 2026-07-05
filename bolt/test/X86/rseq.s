## Check that llvm-bolt avoids optimization of functions referenced from
## __rseq_cs section, i.e. containing critical sections and abort handlers used
## by restartable sequences in tcmalloc.

# RUN: %clang %cflags %s -o %t -nostdlib -no-pie -Wl,-q
# RUN: llvm-bolt %t -o %t.bolt --print-cfg 2>&1 | FileCheck %s
# RUN: %clang %cflags %s -o %t.pie -nostdlib -pie -Wl,-q
# RUN: llvm-bolt %t.pie -o %t.pie.bolt 2>&1 | FileCheck %s

# CHECK: restartable sequence reference detected in _start
# CHECK: restartable sequence reference detected in __rseq_abort

## Force relocations against .text
  .text
.reloc 0, R_X86_64_NONE

  .global _start
  .type _start, %function
_start:
  pushq %rbp
  mov %rsp, %rbp
.L1:
  pop %rbp
.L2:
  retq
  .size _start, .-_start

  .section __rseq_abort, "ax"
## Signature for rseq abort IP. Unmarked in the symbol table.
  .byte 0x0f, 0x1f, 0x05
  .long 0x42424242
.L3:
  jmp .L2

.section __rseq_cs, "aw"
.balign 32
  .quad .L1
  .quad .L3
