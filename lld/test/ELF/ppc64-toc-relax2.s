# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: echo 'addis 5, 2, .LC0@toc@ha; ld 5, .LC0@toc@l(5); foo: \
# RUN:   .section .toc,"aw",@progbits; .LC0: .tc foo[TC], foo' \
# RUN:   | llvm-mc -filetype=obj -triple=powerpc64le - -o %t1.o
# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK-LABEL: <_start>:
.globl _start
_start:
## Perform toc-indirect to toc-relative relaxation even if there are unrelated instructions in between.
# CHECK-NEXT:   addis 3, 2, -2
# CHECK-NEXT:   li 9, 0
# CHECK-NEXT:   addi 3, 3, 32752
# CHECK-NEXT:   lwa 3, 0(3)
  addis 3, 2, .LC1@toc@ha  # R_PPC64_TOC16_HA
  li    9, 0
  ld    3, .LC1@toc@l(3)   # R_PPC64_TOC16_LO_DS
  lwa   3, 0(3)

## R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS can interleave.
# CHECK-NEXT:   addis 3, 2, -2
# CHECK-NEXT:   addis 4, 2, -2
# CHECK-NEXT:   addi 3, 3, 32752
# CHECK-NEXT:   addi 4, 4, 32756
  addis 3, 2, .LC1@toc@ha
  addis 4, 2, .LC2@toc@ha
  ld    3, .LC1@toc@l(3)
  ld    4, .LC2@toc@l(4)

## We choose to be conservative: the presence of R_PPC64_TOC16_LO
## suppresses relaxation for the symbol.
## R_PPC64_TOC16_HA and R_PPC64_TOC16_LO_DS pairs are not relaxed as well.
# CHECK-NEXT:   nop
# CHECK-NEXT:   addi 3, 2, -32768
# CHECK-NEXT:   li 9, 0
# CHECK-NEXT:   nop
# CHECK-NEXT:   ld 4, -32768(2)
  addis 3, 2, .LC0@toc@ha  # R_PPC64_TOC16_HA
  addi  3, 3, .LC0@toc@l   # R_PPC64_TOC16_LO
  li    9, 0
  addis 4, 2, .LC0@toc@ha
  ld    4, .LC0@toc@l(4)

# CHECK-COUNT-3:   blr
AES_encrypt:
  blr
AES_decrypt:
  blr
BN_free:
  blr

## %t1.o has relaxable relocation pairs referencing its .toc which is different
## from %t.o(.toc). The suppression in %t.o does not affect %t1.o even if
## the relocation addends are the same.
# CHECK-NEXT:   addis 5, 2, -1
# CHECK-NEXT:   addi 5, 5, -32768

.section .toc,"aw",@progbits
.LC0:
  .tc AES_encrypt[TC], AES_encrypt
.LC1:
  .tc AES_decrypt[TC], AES_decrypt
.LC2:
  .tc BN_free[TC], BN_free
