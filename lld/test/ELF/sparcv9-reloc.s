# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
# RUN: ld.lld %t.o --defsym=a=0x0123456789ABCDEF --defsym=b=0x0123456789A --defsym=c=0x01234567 -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-objdump -s %t | FileCheck --check-prefix=HEX %s

## R_SPARC_HH22, R_SPARC_HM10
# CHECK-LABEL: section .ABS_64:
# CHECK:        sethi 18641, %o0
# CHECK-NEXT:   or %o0, 359, %o0
.section .ABS_64,"ax",@progbits
  sethi %hh(a), %o0
  or    %o0, %hm(a), %o0

## R_SPARC_H44, R_SPARC_M44, R_SPARC_L44
# CHECK-LABEL: section .ABS_44:
# CHECK:        sethi 18641, %o0
# CHECK:        or %o0, 359, %o0
# CHECK:        or %o0, 2202, %o0
.section .ABS_44,"ax",@progbits
  sethi %h44(b), %o0
  or    %o0, %m44(b), %o0
  sllx  %o0, 12, %o0
  or    %o0, %l44(b), %o0

## R_SPARC_HI22, R_SPARC_LO10
# CHECK-LABEL: section .ABS_32:
# CHECK:        sethi 18641, %o0
# CHECK-NEXT:   or %o0, 359, %o0
.section .ABS_32,"ax",@progbits
  sethi %hi(c), %o0
  or    %o0, %lo(c), %o0

## R_SPARC_64, R_SPARC_32
# HEX-LABEL: section .ABS_DATA:
# HEX-NEXT:  01234567 89abcdef 01234567
.section .ABS_DATA,"ax",@progbits
  .quad a
  .long c
