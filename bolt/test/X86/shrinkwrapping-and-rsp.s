## This checks that shrink wrapping does attempt at accessing stack elements
## using RSP when the function is aligning RSP and changing offsets.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata \
# RUN:     --frame-opt=all --simplify-conditional-tail-calls=false \
# RUN:     --eliminate-unreachable=false | FileCheck %s

## Here we have a function that aligns the stack at prologue. Stack pointer
## analysis can't try to infer offset positions after AND because that depends
## on the runtime value of the stack pointer of callee (whether it is misaligned
## or not).
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 1
  push  %rbp
  mov   %rsp, %rbp
  push  %rbx
  push  %r14
  and    $0xffffffffffffffe0,%rsp
  subq  $0x20, %rsp
b:  je  hot_path
# FDATA: 1 _start #b# 1 _start #hot_path# 0 1
cold_path:
  mov %r14, %rdi
  mov %rbx, %rdi
  # Block push-pop mode by using an instruction that requires the
  # stack to be aligned to 128B. This will force the pass
  # to try to index stack elements by using RSP +offset directly, but
  # we do not know how to access individual elements of the stack thanks
  # to the alignment.
  movdqa	%xmm8, (%rsp)
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
hot_path:
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
  .cfi_endproc
  .size _start, .-_start

# CHECK:   BOLT-INFO: Shrink wrapping moved 0 spills inserting load/stores and 0 spills inserting push/pops
