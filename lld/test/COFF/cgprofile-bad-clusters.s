# REQUIRES: x86
# This test checks that CallGraphSort ignores edges that would form "bad"
# clusters.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t
# RUN: echo "A C 1" > %t.call_graph
# RUN: echo "E B 4" >> %t.call_graph
# RUN: echo "C D 2" >> %t.call_graph
# RUN: echo "B D 1" >> %t.call_graph
# RUN: echo "F G 6" >> %t.call_graph
# RUN: echo "G H 5" >> %t.call_graph
# RUN: echo "H I 4" >> %t.call_graph
# RUN: lld-link /subsystem:console /entry:A %t /call-graph-ordering-file:%t.call_graph /out:%t2 /debug:symtab
# RUN: llvm-nm --numeric-sort %t2 | FileCheck %s

    .section    .text,"ax",one_only,A
    .globl A
A:
    retq

    .section    .text,"ax",one_only,D
D:
    .fill 1000, 1, 0

    .section    .text,"ax",one_only,E
E:
    retq

    .section    .text,"ax",one_only,C
C:
    retq

    .section    .text,"ax",one_only,B
B:
    .fill 1000, 1, 0

    .section    .text,"ax",one_only,F
F:
    .fill (1024 * 1024) - 1, 1, 0

    .section    .text,"ax",one_only,G
G:
    retq

    .section    .text,"ax",one_only,H
H:
    retq

    .section    .text,"ax",one_only,I
I:
    .fill 13, 1, 0

# CHECK: 140001000 t H
# CHECK: 140001001 t I
# CHECK: 14000100e T A
# CHECK: 14000100f t C
# CHECK: 140001010 t E
# CHECK: 140001011 t B
# CHECK: 1400013f9 t D
# CHECK: 1400017e1 t F
# CHECK: 1401017e0 t G
