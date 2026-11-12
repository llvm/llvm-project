# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:  . = SIZEOF_HEADERS; \
# RUN:  .keep : { KEEP( INPUT_SECTION_FLAGS(!SHF_WRITE) *(.sec*)) } \
# RUN:  }" > %t.script
# RUN: ld.lld --gc-sections -o %t --script %t.script %t.o
# RUN: llvm-readobj --symbols %t | FileCheck %s

## Check that INPUT_SECTION_FLAGS can be used within KEEP, and affects what
## is kept.
# CHECK: Name: keep
# CHECK-NOT: NAME: collect
.text
.global _start
_start:
 .long 0

.section .sec1, "a"
.global keep
keep:
 .long 1

.section .sec2, "aw"
.global collect
collect:
 .long 2
