# This test checks that remove double jumps pass works properly with
# non-local branches.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostartfiles -nodefaultlibs %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt -peepholes=double-jumps

  .text
  .align 4
  .global dummy1
  .type dummy1, %function
dummy1:
  mov x2, x0
  ret
  .size dummy1, .-dummy1

  .global dummy2
  .type dummy2, %function
dummy2:
  mov x1, x0
  ret
  .size dummy2, .-dummy2

  .global _start
  .type _start, %function
_start:
  cbz  x10, 1f
  b dummy1
1:
  b dummy2
  .size _start, .-_start
