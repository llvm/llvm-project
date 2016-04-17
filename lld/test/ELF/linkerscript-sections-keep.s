# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

## First check that section "keep" is garbage collected without using KEEP
# RUN: echo "SECTIONS { \
# RUN:  .text : { *(.text) } \
# RUN:  .keep : { *(.keep) } \
# RUN:  .temp : { *(.temp) }}" > %t.script
# RUN: ld.lld --gc-sections -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | \
# RUN:   FileCheck -check-prefix=SECGC %s
# SECGC:      Sections:
# SECGC-NEXT: Idx Name          Size      Address          Type
# SECGC-NEXT:   0               00000000 0000000000000000
# SECGC-NEXT:   1 .text         00000007 0000000000000158 TEXT DATA
# SECGC-NEXT:   2 .temp         00000004 000000000000015f DATA

## Now apply KEEP command to preserve the section.
# RUN: echo "SECTIONS { \
# RUN:  .text : { *(.text) } \
# RUN:  .keep : { KEEP(*(.keep)) } \
# RUN:  .temp : { *(.temp) }}" > %t.script
# RUN: ld.lld --gc-sections -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | \
# RUN:   FileCheck -check-prefix=SECNOGC %s
# SECNOGC:      Sections:
# SECNOGC-NEXT: Idx Name          Size      Address          Type
# SECNOGC-NEXT:   0               00000000 0000000000000000
# SECNOGC-NEXT:   1 .text         00000007 0000000000000158 TEXT DATA
# SECNOGC-NEXT:   2 .keep         00000004 000000000000015f DATA
# SECNOGC-NEXT:   3 .temp         00000004 0000000000000163 DATA

## A section name matches two entries in the SECTIONS directive. The
## first one doesn't have KEEP, the second one does. If section that have
## KEEP is the first in order then section is NOT collected.
# RUN: echo "SECTIONS { \
# RUN:  .keep : { KEEP(*(.keep)) } \
# RUN:  .nokeep : { *(.keep) }}" > %t.script
# RUN: ld.lld --gc-sections -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | \
# RUN:   FileCheck -check-prefix=KEEP-AT-FIRST %s
# KEEP-AT-FIRST:      Sections:
# KEEP-AT-FIRST-NEXT:  Idx Name          Size      Address         Type
# KEEP-AT-FIRST-NEXT:   0               00000000 0000000000000000
# KEEP-AT-FIRST-NEXT:   1 .keep         00000004 0000000000000120 DATA
# KEEP-AT-FIRST-NEXT:   2 .temp         00000004 0000000000000124 DATA
# KEEP-AT-FIRST-NEXT:   3 .text         00000007 0000000000000128 TEXT DATA
# KEEP-AT-FIRST-NEXT:   4 .symtab       00000060 0000000000000000
# KEEP-AT-FIRST-NEXT:   5 .shstrtab     0000002d 0000000000000000
# KEEP-AT-FIRST-NEXT:   6 .strtab       00000012 0000000000000000

## The same, but now section without KEEP is at first place.
## It will be collected then.
## This test checks that lld behavior is equal to gold linker.
## ld.bfd has different behavior, it prevents the section .keep
## from collecting in this case either.
# RUN: echo "SECTIONS { \
# RUN:  .nokeep : { *(.keep) } \
# RUN:  .keep : { KEEP(*(.keep)) }}" > %t.script
# RUN: ld.lld --gc-sections -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | \
# RUN:   FileCheck -check-prefix=KEEP-AT-SECOND %s
# KEEP-AT-SECOND:      Sections:
# KEEP-AT-SECOND-NEXT:  Idx Name          Size      Address         Type
# KEEP-AT-SECOND-NEXT:   0               00000000 0000000000000000
# KEEP-AT-SECOND-NEXT:   1 .temp         00000004 0000000000000120 DATA
# KEEP-AT-SECOND-NEXT:   2 .text         00000007 0000000000000124 TEXT DATA
# KEEP-AT-SECOND-NEXT:   3 .symtab       00000048 0000000000000000
# KEEP-AT-SECOND-NEXT:   4 .shstrtab     00000027 0000000000000000
# KEEP-AT-SECOND-NEXT:   5 .strtab       0000000d 0000000000000000

.global _start
_start:
 mov temp, %eax

.section .keep, "a"
keep:
 .long 1

.section .temp, "a"
temp:
 .long 2
