# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

## Default case: abc and abx included in text.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.abc .abx) } }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -section-headers %t.out | \
# RUN:   FileCheck -check-prefix=SEC-DEFAULT %s
# SEC-DEFAULT:      Sections:
# SEC-DEFAULT-NEXT: Idx Name          Size      Address          Type
# SEC-DEFAULT-NEXT:   0               00000000 0000000000000000
# SEC-DEFAULT-NEXT:   1 .text         00000008 0000000000011000 TEXT DATA
# SEC-DEFAULT-NEXT:   2 .abcd         00000004 0000000000011008 TEXT DATA
# SEC-DEFAULT-NEXT:   3 .ad           00000004 000000000001100c TEXT DATA 
# SEC-DEFAULT-NEXT:   4 .ag           00000004 0000000000011010 TEXT DATA 
# SEC-DEFAULT-NEXT:   5 .symtab       00000030 0000000000000000 
# SEC-DEFAULT-NEXT:   6 .shstrtab     0000002f 0000000000000000 
# SEC-DEFAULT-NEXT:   7 .strtab       00000008 0000000000000000 

## Now replace the symbol with '?' and check that results are the same.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.abc .ab?) } }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -section-headers %t.out | \
# RUN:   FileCheck -check-prefix=SEC-DEFAULT %s

## Now see how replacing '?' with '*' will consume whole abcd.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.abc .ab*) } }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -section-headers %t.out | \
# RUN:   FileCheck -check-prefix=SEC-ALL %s
# SEC-ALL:      Sections:
# SEC-ALL-NEXT: Idx Name          Size      Address          Type
# SEC-ALL-NEXT:   0               00000000 0000000000000000
# SEC-ALL-NEXT:   1 .text         0000000c 0000000000011000 TEXT DATA
# SEC-ALL-NEXT:   2 .ad           00000004 000000000001100c TEXT DATA 
# SEC-ALL-NEXT:   3 .ag           00000004 0000000000011010 TEXT DATA 
# SEC-ALL-NEXT:   4 .symtab       00000030 0000000000000000 
# SEC-ALL-NEXT:   5 .shstrtab     00000029 0000000000000000 
# SEC-ALL-NEXT:   6 .strtab       00000008 0000000000000000 

## All sections started with .a are merged.
# RUN: echo "SECTIONS { \
# RUN:      .text : { *(.a*) } }" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t
# RUN: llvm-objdump -section-headers %t.out | \
# RUN:   FileCheck -check-prefix=SEC-NO %s
# SEC-NO: Sections:
# SEC-NO-NEXT: Idx Name          Size      Address          Type
# SEC-NO-NEXT:   0               00000000 0000000000000000 
# SEC-NO-NEXT:   1 .text         00000014 0000000000011000 TEXT DATA 
# SEC-NO-NEXT:   2 .symtab       00000030 0000000000000000 
# SEC-NO-NEXT:   3 .shstrtab     00000021 0000000000000000 
# SEC-NO-NEXT:   4 .strtab       00000008 0000000000000000 

.text
.section .abc,"ax",@progbits
.long 0

.text
.section .abx,"ax",@progbits
.long 0

.text
.section .abcd,"ax",@progbits
.long 0

.text
.section .ad,"ax",@progbits
.long 0

.text
.section .ag,"ax",@progbits
.long 0


.globl _start
_start:
