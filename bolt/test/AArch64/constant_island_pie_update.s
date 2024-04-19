// This test checks that the constant island offset and value is updated if
// it has dynamic relocation. The tests checks both .rela.dyn and relr.dyn
// dynamic relocations.
// Also check that we don't duplicate CI if it has dynamic relocations.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
// .rela.dyn
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.rela.exe -nostdlib \
# RUN:   -Wl,-q -Wl,-z,notext
# RUN: llvm-bolt %t.rela.exe -o %t.rela.bolt --use-old-text=0 --lite=0
# RUN: llvm-objdump -j .text -d --show-all-symbols %t.rela.bolt | FileCheck %s
# RUN: llvm-readelf -rsW %t.rela.bolt | FileCheck --check-prefix=ELFCHECK %s
// .relr.dyn
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.relr.exe -nostdlib \
# RUN:   -Wl,-q -Wl,-z,notext -Wl,--pack-dyn-relocs=relr
# RUN: llvm-objcopy --remove-section .rela.mytext %t.relr.exe
# RUN: llvm-bolt %t.relr.exe -o %t.relr.bolt --use-old-text=0 --lite=0
# RUN: llvm-objdump -j .text -d --show-all-symbols %t.relr.bolt | FileCheck %s
# RUN: llvm-objdump -j .text -d %t.relr.bolt | \
# RUN:   FileCheck %s --check-prefix=ADDENDCHECK
# RUN: llvm-readelf -rsW %t.relr.bolt | FileCheck --check-prefix=RELRELFCHECK %s
# RUN: llvm-readelf -SW %t.relr.bolt | FileCheck --check-prefix=RELRSZCHECK %s

// Check that the CI value was updated
# CHECK: [[#%x,ADDR:]] <exitLocal>:
# CHECK: {{.*}} <$d>:
# CHECK-NEXT: {{.*}} .word 0x{{[0]+}}[[#ADDR]]
# CHECK-NEXT: {{.*}} .word 0x00000000
# CHECK-NEXT: {{.*}} .word 0x{{[0]+}}[[#ADDR]]
# CHECK-NEXT: {{.*}} .word 0x00000000
# CHECK-NEXT: {{.*}} .word 0x00000000
# CHECK-NEXT: {{.*}} .word 0x00000000
# CHECK-NEXT: {{.*}} .word 0x{{[0]+}}[[#ADDR]]
# CHECK-NEXT: {{.*}} .word 0x00000000

// Check that addend was properly patched in mytextP with stripped relocations
# ADDENDCHECK: [[#%x,ADDR:]] <exitLocal>:
# ADDENDCHECK: {{.*}} <mytextP>:
# ADDENDCHECK-NEXT: {{.*}} .word 0x{{[0]+}}[[#ADDR]]
# ADDENDCHECK-NEXT: {{.*}} .word 0x00000000

// Check that we've relaxed adr to adrp + add to refer external CI
# CHECK: <addressDynCi>:
# CHECK-NEXT: adrp
# CHECK-NEXT: add

// Check that relocation offsets were updated
# ELFCHECK: [[#%x,OFF:]] [[#%x,INFO_DYN:]] R_AARCH64_RELATIVE
# ELFCHECK-NEXT: [[#OFF + 8]] {{0*}}[[#INFO_DYN]] R_AARCH64_RELATIVE
# ELFCHECK-NEXT: [[#OFF + 24]] {{0*}}[[#INFO_DYN]] R_AARCH64_RELATIVE
# ELFCHECK-NEXT: {{.*}} R_AARCH64_RELATIVE
# ELFCHECK: {{.*}}[[#OFF]] {{.*}} $d

# RELRELFCHECK:       $d{{$}}
# RELRELFCHECK-NEXT:  $d + 0x8{{$}}
# RELRELFCHECK-NEXT:  $d + 0x18{{$}}
# RELRELFCHECK-NEXT:  mytextP
# RELRELFCHECK-EMPTY:

// Check that .relr.dyn size is 2 bytes to ensure that last 3 relocations were
// encoded as a bitmap so the total section size for 3 relocations is 2 bytes.
# RELRSZCHECK: .relr.dyn RELR [[#%x,ADDR:]] [[#%x,OFF:]] {{0*}}10

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
  .xword exitLocal
  .xword 0
  .xword exitLocal
  .size _start, .-_start

  .global addressDynCi
  .type addressDynCi, %function
addressDynCi:
  adr x1, .Lci
  bl _start
.size addressDynCi, .-addressDynCi

  .section ".mytext", "ax"
  .balign 8
  .global dummy
  .type dummy, %function
dummy:
  nop
  .word 0
  .size dummy, .-dummy

  .global mytextP
mytextP:
  .xword exitLocal
  .size mytextP, .-mytextP
