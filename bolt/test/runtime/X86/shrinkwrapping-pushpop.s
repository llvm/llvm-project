# This reproduces a bug with shrink wrapping when trying to move
# push-pops where restores were not validated to be POPs (and could be
# a regular load, which violated our assumptions). Check that we
# identify those cases.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# Delete our BB symbols so BOLT doesn't mark them as entry points
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --relocs=1 --frame-opt=all --print-finalized \
# RUN:    --print-only=main --data %t.fdata -o %t.out | FileCheck %s

# RUN: %t.out

# CHECK: BOLT-INFO: Shrink wrapping moved 1 spills inserting load/stores and 0 spills inserting push/pops

  .text
  .globl  main
  .type main, %function
  .p2align  4
main:
# FDATA: 0 [unknown] 0 1 main 0 0 510
  pushq %rbp
  movq  %rsp, %rbp
  pushq %rbx                  # We save rbx here, but there is an
                              # opportunity to move it to .LBB2
  subq  $0x18, %rsp
  cmpl  $0x2, %edi
.J1:
  jb    .BBend
# FDATA: 1 main #.J1# 1 main #.BB2# 0 10
# FDATA: 1 main #.J1# 1 main #.BBend# 0 500
.BB2:
  movq $2, %rbx               # Use rbx in a cold block
  xorq %rax, %rax
  movb mystring, %al
  addq %rbx, %rax
  movb %al, mystring
  leaq mystring, %rdi
# Avoid making the actual call here to allow push-pop mode to operate
# without fear of an unknown function that may require aligned stack
#  callq puts

.BBend:
  mov -0x08(%rbp), %rbx       # Original restore is here. Instead of a pop
                              # we use a load to reproduce gcc behavior while
                              # using leave in epilogue. Ordinarily it would
                              # addq $0x18, %rsp followed by pop %rbx
  xorq %rax, %rax
  leaveq
  retq
  .size main, .-main

  .data
mystring: .asciz "0 is rbx mod 10 contents in decimal\n"
