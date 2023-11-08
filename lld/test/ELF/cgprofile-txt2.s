# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "A B 5" > %t.call_graph
# RUN: echo "B C 50" >> %t.call_graph
# RUN: echo "C D 40" >> %t.call_graph
# RUN: echo "D B 10" >> %t.call_graph
# RUN: echo "D E 1" >> %t.call_graph
# RUN: ld.lld -e A %t --call-graph-ordering-file %t.call_graph --call-graph-profile-sort=hfsort -o %t2
# RUN: llvm-readobj --symbols %t2 | FileCheck %s --check-prefix=CHECKC3
# RUN: ld.lld -e A %t --call-graph-ordering-file %t.call_graph --call-graph-profile-sort=cdsort -o %t2
# RUN: llvm-readobj --symbols %t2 | FileCheck %s --check-prefix=CHECKCDS

## The expected order is [B, C, D, E, A]
# CHECKC3:      Name: A
# CHECKC3-NEXT: Value: 0x201123
# CHECKC3:      Name: B
# CHECKC3-NEXT: Value: 0x201120
# CHECKC3:      Name: C
# CHECKC3-NEXT: Value: 0x201121
# CHECKC3:      Name: D
# CHECKC3-NEXT: Value: 0x201122
# CHECKC3:      Name: E
# CHECKC3-NEXT: Value: 0x201123

## The expected order is [A, B, C, D, E]
# CHECKCDS:      Name: A
# CHECKCDS-NEXT: Value: 0x201120
# CHECKCDS:      Name: B
# CHECKCDS-NEXT: Value: 0x201121
# CHECKCDS:      Name: C
# CHECKCDS-NEXT: Value: 0x201122
# CHECKCDS:      Name: D
# CHECKCDS-NEXT: Value: 0x201123
# CHECKCDS:      Name: E
# CHECKCDS-NEXT: Value: 0x201124

.section    .text.A,"ax",@progbits
.globl  A
A:
 nop

.section    .text.B,"ax",@progbits
.globl  B
B:
 nop

.section    .text.C,"ax",@progbits
.globl  C
C:
 nop

.section    .text.D,"ax",@progbits
.globl  D
D:
 nop

.section    .text.E,"ax",@progbits
.globl  E
E:
