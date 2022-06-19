// This test checks that the constant island value is updated if it
// has dynamic relocation.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -Wl,-q -nostdlib -Wl,-z,notext
# RUN: llvm-bolt %t.exe -o %t.bolt --use-old-text=0 --lite=0
# RUN: llvm-objdump -j .text -dR %t.bolt | FileCheck %s

# CHECK: R_AARCH64_RELATIVE *ABS*+0x[[#%x,ADDR:]]
# CHECK: [[#ADDR]] <exitLocal>:
# CHECK: {{.*}} <$d>:
# CHECK-NEXT: {{.*}} .word 0x{{[0]+}}[[#ADDR]]
# CHECK-NEXT: {{.*}} .word 0x00000000

  .text
  .align 4
  .local exitLocal
  .type exitLocal, %function
exitLocal:
  add x1, x1, #1
  add x1, x1, #1
  ret
  .size exitLocal, .-exitLocal

  .global _start
  .type _start, %function
_start:
  mov x0, #0
  adr x1, .Lci
  ldr x1, [x1]
  blr x1
  mov x0, #1
  bl exitLocal
  nop
.Lci:
  .xword exitLocal
  .size _start, .-_start
