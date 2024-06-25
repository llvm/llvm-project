# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: not --crash llvm-bolt -instrument -instrumentation-sleep-time=1 %t.exe \
# RUN:  -o %t.instr 2>&1 | FileCheck %s

# CHECK: not implemented

  .text
  .align 4
  .global _start
  .type _start, %function
_start:
   bl foo
   ret
  .size _start, .-_start

  .global foo
  .type foo, %function
foo:
  mov	w0, wzr
  ret
  .size foo, .-foo
