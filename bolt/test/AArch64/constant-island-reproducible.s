# This test checks that the sequence of generated constant islands is the same across
# every llvm-bolt run.
# The 1 KB alignment is used to place the main and dummy0 functions far enough apart.
# If the functions are close, the original constant islands are used instead of
# the generated ones (no binary difference in this case).

# REQUIRES: system-linux

# RUN: %clang %s %cflags -no-pie -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0
# RUN: llvm-objdump --disassemble-symbols=main %t.bolt | FileCheck %s

# CHECK: 00000010 udf #0x10
# CHECK: 00000020 udf #0x20
# CHECK: 00000030 udf #0x30
# CHECK: 00000040 udf #0x40

  .text
  .align 10
  .global main
  .type main, %function
main:
  adr x0, CI0
  ldr x0, [x0]
  adr x0, CI1
  ldr x0, [x0]
  adr x0, CI2
  ldr x0, [x0]
  adr x0, CI3
  ldr x0, [x0]
  ret
  .size main, .-main

  .align 10
  .global dummy0
  .type dummy0, %function
dummy0:
  mov x0, #0
  ret
  .size dummy0, .-dummy0
CI0:
  .xword 0x10
CI1:
  .xword 0x20

  .align 4
  .global dummy1
  .type dummy1, %function
dummy1:
  mov x0, #0
  ret
  .size dummy1, .-dummy1
CI2:
  .xword 0x30
CI3:
  .xword 0x40
