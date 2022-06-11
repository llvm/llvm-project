# This reproduces a bug triggered by a relocation referencing a section symbol
# plus a negative reloc. BOLT handles such cases specially, but while doing so,
# it was failing to sign extend a negative result for the relocation (encoded
# in the immediate operand of an LEA instruction).
# Originally triggered by https://fossies.org/linux/glib/glib/guniprop.c
# Line 550: const gchar *p = special_case_table + val - 0x1000000;

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# Delete our BB symbols so BOLT doesn't mark them as entry points
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --relocs=1 --print-finalized --print-only=main -o %t.out

# RUN: %t.out 1 2

  .text
  .globl  main
  .type main, %function
  .p2align  4
main:
  pushq %rbp
  movq  %rsp, %rbp
  subq  $0x18, %rsp
  cmpl  $0x2, %edi
  jb    .BBend
.BB2:
  leaq .data-0x1000000, %rsi     # Use a large negative addend to cause a
                                 # negative result to be encoded in LEA
  addq $0x1000000, %rsi          # Eventually program logic compensates to get
                                 # a real address
  movq $2, %rbx
  xorq %rax, %rax
  movb (%rsi), %al
  addq %rbx, %rax
  movb %al, (%rsi)
  leaq mystring, %rdi
  callq puts

.BBend:
  xorq %rax, %rax
  leaveq
  retq
  .size main, .-main

  .data
mystring: .asciz "0 is rbx mod 10 contents in decimal\n"
