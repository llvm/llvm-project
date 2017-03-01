# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-linux %s -o %t.o
# RUN: echo "SECTIONS { .data 0x4000 : { *(.data) } .text 0x2000 : { *(.text) } }" > %t.script
# RUN: not ld.lld -o %t.so --script %t.script %t.o -shared 2>&1 | FileCheck %s

# CHECK:  error: {{.*}}.script:1: unable to move location counter backward

.quad 0
.data
.quad 0
