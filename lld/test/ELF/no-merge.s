# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { .data : {*(.data.*)} }" > %t0.script
# RUN: ld.lld %t.o -o %t0.out --script %t0.script
# RUN: llvm-objdump -s %t0.out | FileCheck %s

# RUN: ld.lld -O0 %t.o -o %t1.out --script %t0.script
# RUN: llvm-objdump -s %t1.out | FileCheck %s
# CHECK:      Contents of section .data:
# CHECK-NEXT:   0000 01610003

.section .data.aw,"aw",@progbits
.byte 1

.section .data.ams,"aMS",@progbits,1
.asciz "a"

.section .data.am,"aM",@progbits,1
.byte 3
