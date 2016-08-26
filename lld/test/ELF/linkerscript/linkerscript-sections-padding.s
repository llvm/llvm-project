# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

## Check that padding value works:
# RUN: echo "SECTIONS { .mysec : { *(.mysec*) } =0x112233445566778899 }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -s %t.out | FileCheck -check-prefix=YES %s
# YES: 0120  66223344 55667788 99112233 44556677

## Confirming that address was correct:
# RUN: echo "SECTIONS { .mysec : { *(.mysec*) } =0x998877665544332211 }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -s %t.out | FileCheck -check-prefix=YES2 %s
# YES2: 0120  66887766 55443322 11998877 66554433

## Default padding value is 0x00:
# RUN: echo "SECTIONS { .mysec : { *(.mysec*) } }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -s %t.out | FileCheck -check-prefix=NO %s
# NO: 0120  66000000 00000000 00000000 00000000

## Decimal value.
# RUN: echo "SECTIONS { .mysec : { *(.mysec*) } =777 }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -s %t.out | FileCheck -check-prefix=DEC %s
# DEC: 0120  66000309 00000309 00000309 00000309

## Invalid hex value:
# RUN: echo "SECTIONS { .mysec : { *(.mysec*) } =0x99XX }" > %t.script
# RUN: not ld.lld -o %t.out --script %t.script %t 2>&1 \
# RUN:   | FileCheck --check-prefix=ERR2 %s
# ERR2: not a hexadecimal value: XX

.section        .mysec.1,"a"
.align  16
.byte   0x66

.section        .mysec.2,"a"
.align  16
.byte   0x66

.globl _start
_start:
 nop
