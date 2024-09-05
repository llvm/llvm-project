## This test checks that POPF will not crash our frame analysis pass

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata --frame-opt=all --lite=0


  .globl _start
_start:
    .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 6
    je a
b:  jne _start
# FDATA: 1 _start #b# 1 _start #c# 0 3

c:
    pushf
    push  %rbx
    push  %rbp
    addq $0x50, -0x30(%rbp)
    pop   %rbp
    pop   %rbx
    popf

## This basic block is treated as having 0 execution count.
a:
    ud2
    .cfi_endproc
