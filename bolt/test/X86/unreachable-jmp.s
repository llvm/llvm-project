## This checks that we don't create an invalid CFG when there is an
## unreachable direct jump right after an indirect one.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata \
# RUN:     --eliminate-unreachable --print-cfg | FileCheck %s

  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 1
  push  %rbp
  mov   %rsp, %rbp
  push  %rbx
  push  %r14
  subq  $0x20, %rsp
  movq  %rdi, %rcx
b:
  jmpq  *JUMP_TABLE(,%rcx,8)
# FDATA: 1 _start #b# 1 _start #hotpath# 0 20
## Unreachable direct jump here. Our CFG should still make sense and properly
## place this instruction in a new basic block.
  jmp  .lbb2
.lbb1:  je  .lexit
.lbb2:
  xorq  %rax, %rax
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
hotpath:
  movq  $2, %rax
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
.lexit:
  movq  $1, %rax
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
  .cfi_endproc
  .size _start, .-_start

  .rodata
  .globl JUMP_TABLE
JUMP_TABLE:
  .quad .lbb1
  .quad .lbb2
  .quad hotpath

## No basic blocks above should have 4 successors! That is a bug.
# CHECK-NOT:   Successors: {{.*}} (mispreds: 0, count: 20), {{.*}} (mispreds: 0, count: 0), {{.*}} (mispreds: 0, count: 0), {{.*}} (mispreds: 0, count: 0)
# Check successful removal of stray direct jmp
#  CHECK:  UCE removed 1 block
