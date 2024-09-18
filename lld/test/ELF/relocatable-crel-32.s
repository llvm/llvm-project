# REQUIRES: ppc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=powerpc -crel a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=powerpc -crel b.s -o b.o
# RUN: ld.lld -r b.o a.o -o out
# RUN: llvm-readobj -r out | FileCheck %s --check-prefixes=CHECK,CRELFOO

# RUN: llvm-mc -filetype=obj -triple=powerpc a.s -o a1.o
# RUN: ld.lld -r b.o a1.o -o out1
# RUN: llvm-readobj -r out1 | FileCheck %s --check-prefixes=CHECK,RELAFOO
# RUN: ld.lld -r a1.o b.o -o out2
# RUN: llvm-readobj -r out2 | FileCheck %s --check-prefixes=CHECK2

# CHECK:      Relocations [
# CHECK-NEXT:   Section (2) .crel.text {
# CHECK-NEXT:     0x0 R_PPC_REL24 fb 0x0
# CHECK-NEXT:     0x4 R_PPC_REL24 foo 0x0
# CHECK-NEXT:     0x8 R_PPC_REL24 .text.foo 0x0
# CHECK-NEXT:     0xE R_PPC_ADDR16_HA .rodata.str1.1 0x4
# CHECK-NEXT:     0x12 R_PPC_ADDR16_LO .rodata.str1.1 0x4
# CHECK-NEXT:     0x16 R_PPC_ADDR16_HA .rodata.str1.1 0x0
# CHECK-NEXT:     0x1A R_PPC_ADDR16_LO .rodata.str1.1 0x0
# CHECK-NEXT:   }
# CRELFOO-NEXT: Section (4) .crel.text.foo {
# RELAFOO-NEXT: Section (4) .rela.text.foo {
# CHECK-NEXT:     0x0 R_PPC_REL24 g 0x0
# CHECK-NEXT:     0x4 R_PPC_REL24 g 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECK2:      Relocations [
# CHECK2-NEXT:   Section (2) .crel.text {
# CHECK2-NEXT:     0x0 R_PPC_REL24 foo 0x0
# CHECK2-NEXT:     0x4 R_PPC_REL24 .text.foo 0x0
# CHECK2-NEXT:     0xA R_PPC_ADDR16_HA .rodata.str1.1 0x4
# CHECK2-NEXT:     0xE R_PPC_ADDR16_LO .rodata.str1.1 0x4
# CHECK2-NEXT:     0x12 R_PPC_ADDR16_HA .rodata.str1.1 0x0
# CHECK2-NEXT:     0x16 R_PPC_ADDR16_LO .rodata.str1.1 0x0
# CHECK2-NEXT:     0x18 R_PPC_REL24 fb 0x0
# CHECK2-NEXT:   }
# CHECK2-NEXT:   Section (4) .rela.text.foo {
# CHECK2-NEXT:     0x0 R_PPC_REL24 g 0x0
# CHECK2-NEXT:     0x4 R_PPC_REL24 g 0x0
# CHECK2-NEXT:   }
# CHECK2-NEXT: ]

#--- a.s
.global _start, foo
_start:
  bl foo
  bl .text.foo
  lis 3, .L.str@ha
  la 3, .L.str@l(3)
  lis 3, .L.str1@ha
  la 3, .L.str1@l(3)

.section .text.foo,"ax"
foo:
  bl g
  bl g

.section .rodata.str1.1,"aMS",@progbits,1
.L.str:
  .asciz  "abc"
.L.str1:
  .asciz  "def"

#--- b.s
.globl fb
fb:
  bl fb
