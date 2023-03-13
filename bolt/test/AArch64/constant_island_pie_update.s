// This test checks that the constant island value is updated if it
// has dynamic relocation.
// Also check that we don't duplicate CI if it has dynamic relocations.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.rela.exe -nostdlib \
# RUN:   -Wl,-q -Wl,-z,notext
# RUN: llvm-bolt %t.rela.exe -o %t.rela.bolt --use-old-text=0 --lite=0
# RUN: llvm-objdump -j .text -d %t.rela.bolt | FileCheck %s
# RUN: llvm-readelf -rsW %t.rela.bolt | FileCheck --check-prefix=ELFCHECK %s

// Check that the CI value was updated
# CHECK: [[#%x,ADDR:]] <exitLocal>:
# CHECK: {{.*}} <$d>:
# CHECK-NEXT: {{.*}} .word 0x{{[0]+}}[[#ADDR]]
# CHECK-NEXT: {{.*}} .word 0x00000000

// Check that we've relaxed adr to adrp + add to refer external CI
# CHECK: <addressDynCi>:
# CHECK-NEXT: adrp
# CHECK-NEXT: add

// Check that relocation offset was updated
# ELFCHECK: [[#%x,OFF:]] [[#%x,INFO_DYN:]] R_AARCH64_RELATIVE
# ELFCHECK: {{.*}}[[#OFF]] {{.*}} $d

  .text
  .align 4
  .local exitLocal
  .type exitLocal, %function
exitLocal:
  add x1, x1, #1
  add x1, x1, #1
  nop
  nop
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

  .global addressDynCi
  .type addressDynCi, %function
addressDynCi:
  adr x1, .Lci
  bl _start
.size addressDynCi, .-addressDynCi
