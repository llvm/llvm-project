# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t
# RUN: echo "A B 5" > %t.call_graph
# RUN: echo "B C 50" >> %t.call_graph
# RUN: echo "C D 40" >> %t.call_graph
# RUN: echo "D B 10" >> %t.call_graph
# RUN: lld-link /subsystem:console /entry:A %t /call-graph-ordering-file:%t.call_graph /out:%t2 /print-symbol-order:%t3
# RUN: FileCheck %s --input-file %t3

# CHECK: B
# CHECK-NEXT: C
# CHECK-NEXT: D
# CHECK-NEXT: A

.section    .text, "x", one_only, A
.globl  A
A:
 nop

.section    .text, "x", one_only, B
.globl  B
B:
 nop

.section    .text, "x", one_only, C
.globl  C
C:
 nop

.section    .text, "x", one_only, D
.globl  D
D:
 nop
