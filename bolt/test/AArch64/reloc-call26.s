## This test checks processing of R_AARCH64_CALL26 relocation
## when option `--funcs` is enabled

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --funcs=func1
# RUN: llvm-objdump -d --disassemble-symbols='_start' %t.bolt | \
# RUN:   FileCheck %s

# CHECK: {{.*}} bl {{.*}} <func1>

  .text
  .align 4
  .global _start
  .type _start, %function
_start:
  bl func1
  mov     w8, #93
  svc     #0
  .size _start, .-_start

  .global func1
  .type func1, %function
func1:
  ret
  .size func1, .-func1
