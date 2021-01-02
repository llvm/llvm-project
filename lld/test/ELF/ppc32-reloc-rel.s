# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.be.o
# RUN: ld.lld %t.be.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpcle %s -o %t.le.o
# RUN: ld.lld %t.le.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.section .R_PPC_REL14,"ax",@progbits
  beq 1f
1:
# CHECK-LABEL: section .R_PPC_REL14:
# CHECK: 100100b4: bt 2, 0x100100b8

.section .R_PPC_REL24,"ax",@progbits
  b 1f
1:
# CHECK-LABEL: section .R_PPC_REL24:
# CHECK: b 0x100100bc

.section .R_PPC_REL32,"ax",@progbits
  .long 1f - .
1:
# HEX-LABEL: section .R_PPC_REL32:
# HEX-NEXT: 10010008 00000004

.section .R_PPC_PLTREL24,"ax",@progbits
  b 1f@PLT+32768
1:
# CHECK-LABEL: section .R_PPC_PLTREL24:
# CHECK: b 0x100100c4

.section .R_PPC_LOCAL24PC,"ax",@progbits
  b 1f@local
1:
# CHECK-LABEL: section .R_PPC_LOCAL24PC:
# CHECK: b 0x100100c8
