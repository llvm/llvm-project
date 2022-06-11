# This reproduces a bug with shrink wrapping when trying to split critical
# edges originating at the same basic block.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe --relocs -o %t.out --data %t.fdata \
# RUN:     --frame-opt=all --simplify-conditional-tail-calls=false \
# RUN:     --eliminate-unreachable=false

  .globl _start
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 1
    push  %rbx
    je  b
c:
    pop %rbx
    ret
b:
    je  f
    jmp *JT(,%rbx,8)
d:
    jmp d
    mov %r14, %rdi
f:
    ret
  .data
JT:
  .quad c
  .quad d
  .quad f
