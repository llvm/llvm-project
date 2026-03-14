# REQUIRES: x86
# Test correctness of call graph ordering.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t
# RUN: lld-link /subsystem:console /entry:A %t /out:%t2 /debug:symtab
# RUN: llvm-nm --numeric-sort %t2 | FileCheck %s --check-prefix=NOSORT 

# RUN: echo "A B 5" > %t.call_graph
# RUN: echo "B C 50" >> %t.call_graph
# RUN: echo "C D 40" >> %t.call_graph
# RUN: echo "D B 10" >> %t.call_graph
# RUN: lld-link /subsystem:console /entry:A %t /call-graph-ordering-file:%t.call_graph /out:%t2 /debug:symtab
# RUN: llvm-nm --numeric-sort %t2 | FileCheck %s 

# NOSORT: 140001000 T A
# NOSORT: 140001001 T B
# NOSORT: 140001002 T C
# NOSORT: 140001003 T D

# CHECK: 140001000 T B
# CHECK: 140001001 T C
# CHECK: 140001002 T D
# CHECK: 140001003 T A

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
