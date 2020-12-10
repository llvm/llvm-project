# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %tfile1.o

# RUN: echo "SECTIONS { .abc : { *(SORT(.foo.*) .a* .a* SORT(.bar.*) .b*) } }" > %t1.script
# RUN: ld.lld -o %t1 --script %t1.script %tfile1.o
# RUN: llvm-readelf -x .abc %t1 | FileCheck %s

## Sections matched by patterns between two SORT are sorted separately by input order.
## Note, GNU ld has a strange behavior with more than one SORT* https://sourceware.org/pipermail/binutils/2020-November/114083.html
## In the absence of SORT, our multi-pattern behavior matches GNU ld.
# CHECK:      Hex dump of section '.abc'
# CHECK-NEXT: 0x00000000 01020306 05040708 090b0c0a

# RUN: echo "SECTIONS { \
# RUN:   .abc : { *(SORT(.foo.* EXCLUDE_FILE (*file1.o) .bar.*) .a* SORT(.bar.*) .b*) } \
# RUN:  }" > %t2.script
# RUN: ld.lld -o %t2 --script %t2.script %tfile1.o
# RUN: llvm-readelf -x .abc %t2 | FileCheck %s

## Non-SORT patterns are sorted by --sort-section, breaking tie by input order.
# RUN: ld.lld -o %t4 --script %t1.script --sort-section=name %tfile1.o
# RUN: llvm-readelf -x .abc %t4 | FileCheck %s --check-prefix=CHECK2

# CHECK2:      Hex dump of section '.abc'
# CHECK2-NEXT: 0x00000000 01020304 05060708 090a0b0c

.text
.globl _start
_start:

.section .foo.2,"a"; .byte 2
.section .foo.3,"a"; .byte 3
.section .foo.1,"a"; .byte 1

.section .a6,"a"; .byte 6
.section .a5,"a"; .byte 5
.section .a4,"a"; .byte 4

.section .bar.7,"a"; .byte 7
.section .bar.9,"a"; .byte 9
.section .bar.8,"a"; .byte 8

.section .b11,"a"; .byte 11
.section .b12,"a"; .byte 12
.section .b10,"a"; .byte 10
