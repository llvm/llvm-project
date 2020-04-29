# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK-LABEL: <_start>:
.globl _start
_start:
## Perform toc-indirect to toc-relative relaxation even if there are unrelated instructions in between.
# CHECK-NEXT:   addis 3, 2, -1
# CHECK-NEXT:   li 9, 0
# CHECK-NEXT:   addi 3, 3, -32768
# CHECK-NEXT:   lwa 3, 0(3)
# CHECK-NEXT:   li 9, 0
  addis 3, 2, .LC0@toc@ha  # R_PPC64_TOC16_HA
  li    9, 0
  ld    3, .LC0@toc@l(3)   # R_PPC64_TOC16_LO_DS
  lwa   3, 0(3)
  li    9, 0

## The R_PPC64_TOC16_HA is not followed by an R_PPC64_TOC16_LO_DS.
## Don't perform toc-indirect to toc-relative relaxation.
# CHECK-NEXT:   nop
# CHECK-NEXT:   addi 3, 2, -32768
# CHECK-NEXT:   blr
  addis 3, 2, .LC0@toc@ha  # R_PPC64_TOC16_HA
  addi  3, 3, .LC0@toc@l   # R_PPC64_TOC16_LO
  blr

AES_encrypt:

.section .toc,"aw",@progbits
.LC0:
  .tc AES_encrypt[TC], AES_encrypt
