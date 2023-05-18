# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## Test the INPUT_SECTION_FLAGS feature. It prefixes an input section list and
## restricts matches to sections that have the required flags and do not have
## any of the must not have flags.

## Uniquely identify each .sec section by flag alone, with .text going into
## to the SHF_EXECINSTR requiring .outsec2
# RUN: echo "SECTIONS { \
# RUN: .outsec1 : { INPUT_SECTION_FLAGS(SHF_ALLOC & !SHF_EXECINSTR & \
# RUN:                                  !SHF_WRITE & !SHF_MERGE) *(.sec.*) } \
# RUN: .outsec2 : { INPUT_SECTION_FLAGS(SHF_ALLOC & SHF_EXECINSTR & !SHF_WRITE\
# RUN:                                  & !SHF_MERGE) *(.sec.* .text) } \
# RUN: .outsec3 : { INPUT_SECTION_FLAGS(SHF_WRITE) *(.sec.*) } \
# RUN: .outsec4 : { INPUT_SECTION_FLAGS(SHF_MERGE & !SHF_STRINGS) *(.sec.*) } \
# RUN: .outsec5 : { INPUT_SECTION_FLAGS(SHF_STRINGS) *(.sec.*) } \
# RUN: } " > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t.o
# RUN: llvm-readobj --symbols %t1 | FileCheck %s
# CHECK:  Name: _start
# CHECK:  Section: .outsec2
# CHECK:  Name: s1
# CHECK:  Section: .outsec1
# CHECK:  Name: s2
# CHECK:  Section: .outsec2
# CHECK:  Name: s3
# CHECK:  Section: .outsec3
# CHECK:  Name: s4
# CHECK:  Section: .outsec4
# CHECK:  Name: s5
# CHECK:  Section: .outsec5

## Same test but using OVERLAY.
# RUN: echo "SECTIONS { \
# RUN: OVERLAY 0x1000 : AT ( 0x4000 ) { \
# RUN: .outsec1 { INPUT_SECTION_FLAGS(SHF_ALLOC & !SHF_EXECINSTR & \
# RUN:                                !SHF_WRITE & !SHF_MERGE) *(.sec.*) }\
# RUN: .outsec2 { INPUT_SECTION_FLAGS(SHF_ALLOC & SHF_EXECINSTR & !SHF_WRITE \
# RUN:                                & !SHF_MERGE) *(.sec.* .text) } \
# RUN: .outsec3 { INPUT_SECTION_FLAGS(SHF_WRITE) *(.sec.*) } \
# RUN: .outsec4 { INPUT_SECTION_FLAGS(SHF_MERGE & !SHF_STRINGS) *(.sec.*) } \
# RUN: .outsec5 { INPUT_SECTION_FLAGS(SHF_STRINGS) *(.sec.*) } \
# RUN: } } " > %t2.script

# RUN: ld.lld -o %t2 --script %t2.script %t.o
# RUN: llvm-readobj --symbols %t2 | FileCheck %s

## Same test but using hex representations of the flags.
# RUN: echo "SECTIONS { \
# RUN: .outsec1 : { INPUT_SECTION_FLAGS(0x2 & !0x4 & !0x1 & !0x10) *(.sec.*) }\
# RUN: .outsec2 : { INPUT_SECTION_FLAGS(0x2 & 0x4 & !0x1 & !0x10) \
# RUN:              *(.sec.* .text) } \
# RUN: .outsec3 : { INPUT_SECTION_FLAGS(0x1) *(.sec.*) } \
# RUN: .outsec4 : { INPUT_SECTION_FLAGS(0x10 & !0x20) *(.sec.*) } \
# RUN: .outsec5 : { INPUT_SECTION_FLAGS(0x20) *(.sec.*) } \
# RUN: } " > %t3.script

# RUN: ld.lld -o %t3 --script %t3.script %t.o
# RUN: llvm-readobj --symbols %t3 | FileCheck %s

## Check that we can handle multiple InputSectionDescriptions in a single
## OutputSection
# RUN: echo "SECTIONS { \
# RUN: .outsec1 : { INPUT_SECTION_FLAGS(SHF_ALLOC & !SHF_EXECINSTR & \
# RUN:                                  !SHF_WRITE & !SHF_MERGE) *(.sec.*) ; \
# RUN:              INPUT_SECTION_FLAGS(SHF_ALLOC & SHF_EXECINSTR & !SHF_WRITE\
# RUN:                                  & !SHF_MERGE)  *(.sec.* *.text) }\
# RUN: } " > %t4.script

# RUN: ld.lld -o %t4 --script %t4.script %t.o
# RUN: llvm-readobj --symbols %t4 | FileCheck --check-prefix MULTIPLE %s

# MULTIPLE:  Name: _start
# MULTIPLE:  Section: .outsec1
# MULTIPLE:  Name: s1
# MULTIPLE:  Section: .outsec1
# MULTIPLE:  Name: s2
# MULTIPLE:  Section: .outsec1
# MULTIPLE:  Name: s3
# MULTIPLE:  Section: .sec.aw
# MULTIPLE:  Name: s4
# MULTIPLE:  Section: .sec.aM
# MULTIPLE:  Name: s5
# MULTIPLE:  Section: .sec.aMS

 .text
 .global _start
_start:
 nop

 .section .sec.a, "a", @progbits
 .globl s1
s1:
 .long 1

 .section .sec.ax, "ax", @progbits
 .globl s2
s2:
 .long 2

 .section .sec.aw, "aw", @progbits
 .globl s3
s3:
 .long 3

 .section .sec.aM, "aM", @progbits, 4
 .globl s4
s4:
 .long 4

 .section .sec.aMS, "aMS", @progbits, 1
 .globl s5
s5:
 .asciz "a"
