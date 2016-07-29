# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   %p/Inputs/linkerscript-sort.s -o %t2.o

# RUN: echo "SECTIONS { .aaa : { *(.aaa.*) } }" > %t1.script
# RUN: ld.lld -o %t1 --script %t1.script %t2.o %t1.o
# RUN: llvm-objdump -s %t1 | FileCheck -check-prefix=UNSORTED %s
# UNSORTED:      Contents of section .aaa:
# UNSORTED-NEXT:   0120 55000000 00000000 11000000 00000000
# UNSORTED-NEXT:   0130 33000000 00000000 22000000 00000000
# UNSORTED-NEXT:   0140 44000000 00000000 05000000 00000000
# UNSORTED-NEXT:   0150 01000000 00000000 03000000 00000000
# UNSORTED-NEXT:   0160 02000000 00000000 04000000 00000000

## Check that SORT works (sorted by name of section).
# RUN: echo "SECTIONS { .aaa : { *(SORT(.aaa.*)) } }" > %t2.script
# RUN: ld.lld -o %t2 --script %t2.script %t2.o %t1.o
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=SORTED_A %s
# SORTED_A:      Contents of section .aaa:
# SORTED_A-NEXT:  0120 11000000 00000000 01000000 00000000
# SORTED_A-NEXT:  0130 22000000 00000000 02000000 00000000
# SORTED_A-NEXT:  0140 33000000 00000000 03000000 00000000
# SORTED_A-NEXT:  0150 44000000 00000000 04000000 00000000
# SORTED_A-NEXT:  0160 55000000 00000000 05000000 00000000

## When we switch the order of files, check that sorting by
## section names is stable.
# RUN: echo "SECTIONS { .aaa : { *(SORT(.aaa.*)) } }" > %t3.script
# RUN: ld.lld -o %t3 --script %t3.script %t1.o %t2.o
# RUN: llvm-objdump -s %t3 | FileCheck -check-prefix=SORTED_B %s
# SORTED_B:      Contents of section .aaa:
# SORTED_B-NEXT:  0120 01000000 00000000 11000000 00000000
# SORTED_B-NEXT:  0130 02000000 00000000 22000000 00000000
# SORTED_B-NEXT:  0140 03000000 00000000 33000000 00000000
# SORTED_B-NEXT:  0150 04000000 00000000 44000000 00000000
# SORTED_B-NEXT:  0160 05000000 00000000 55000000 00000000

## Check that SORT surrounded with KEEP also works.
# RUN: echo "SECTIONS { .aaa : { KEEP (*(SORT(.aaa.*))) } }" > %t3.script
# RUN: ld.lld -o %t3 --script %t3.script %t2.o %t1.o
# RUN: llvm-objdump -s %t3 | FileCheck -check-prefix=SORTED_A %s

.global _start
_start:
 nop

.section .aaa.5, "a"
.quad 5

.section .aaa.1, "a"
.quad 1

.section .aaa.3, "a"
.quad 3

.section .aaa.2, "a"
.quad 2

.section .aaa.4, "a"
.quad 4
