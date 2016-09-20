# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "SECTIONS {.text : {*(.text.*)} end = .;}" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck %s
# CHECK: 00000000000000e9         *ABS*    00000000 end

.global _start
_start:
 nop
