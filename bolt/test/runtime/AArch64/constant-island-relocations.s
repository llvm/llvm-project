# This test checks that the address stored in constant island
# is updated after llvm-bolt

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --use-old-text=0 --lite=0 --trap-old-code
# RUN: %t.bolt

  .text
  .align 4
  .global test
  .type test, %function
test:
  mov x0, #0
  ret
  .size test, .-test

  .global main
  .type main, %function
main:
  adr x0, CI
  ldr x0, [x0]
  br x0
  .size main, .-main
CI:
  .xword test
