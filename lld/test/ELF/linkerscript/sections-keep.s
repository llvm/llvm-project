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
# SECGC-NEXT: Idx Name          Size
# SECGC-NEXT:   0               00000000
# SECGC-NEXT:   1 .text         00000007
# SECGC-NEXT:   2 .temp         00000004

## Now apply KEEP command to preserve the section.
# RUN: echo "SECTIONS { \
# RUN:  .text : { *(.text) } \
# RUN:  .keep : { KEEP(*(.keep)) } \
# RUN:  .temp : { *(.temp) }}" > %t.script
# RUN: ld.lld --gc-sections -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | \
# RUN:   FileCheck -check-prefix=SECNOGC %s
# SECNOGC:      Sections:
# SECNOGC-NEXT: Idx Name          Size
# SECNOGC-NEXT:   0               00000000
# SECNOGC-NEXT:   1 .text         00000007
# SECNOGC-NEXT:   2 .keep         00000004
# SECNOGC-NEXT:   3 .temp         00000004

## A section name matches two entries in the SECTIONS directive. The
## first one doesn't have KEEP, the second one does. If section that have
## KEEP is the first in order then section is NOT collected.
# RUN: echo "SECTIONS { \
# RUN:  .keep : { KEEP(*(.keep)) } \
# RUN:  .nokeep : { *(.keep) }}" > %t.script
# RUN: ld.lld --gc-sections -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | FileCheck -check-prefix=MIXED1 %s
# MIXED1:      Sections:
# MIXED1-NEXT: Idx Name          Size
# MIXED1-NEXT:   0               00000000
# MIXED1-NEXT:   1 .keep         00000004
# MIXED1-NEXT:   2 .temp         00000004
# MIXED1-NEXT:   3 .text         00000007
# MIXED1-NEXT:   4 .symtab       00000060
# MIXED1-NEXT:   5 .shstrtab     0000002d
# MIXED1-NEXT:   6 .strtab       00000012

## The same, but now section without KEEP is at first place.
## gold and bfd linkers disagree here. gold collects .keep while
## bfd keeps it. Our current behavior is compatible with bfd although
## we can choose either way.
# RUN: echo "SECTIONS { \
# RUN:  .nokeep : { *(.keep) } \
# RUN:  .keep : { KEEP(*(.keep)) }}" > %t.script
# RUN: ld.lld --gc-sections -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | FileCheck -check-prefix=MIXED2 %s
# MIXED2:      Sections:
# MIXED2-NEXT: Idx Name          Size
# MIXED2-NEXT:   0               00000000
# MIXED2-NEXT:   1 .nokeep       00000004
# MIXED2-NEXT:   2 .temp         00000004
# MIXED2-NEXT:   3 .text         00000007
# MIXED2-NEXT:   4 .symtab       00000060
# MIXED2-NEXT:   5 .shstrtab     0000002f
# MIXED2-NEXT:   6 .strtab       00000012

.global _start
_start:
 mov temp, %eax

.section .keep, "a"
keep:
 .long 1

.section .temp, "a"
temp:
 .long 2
