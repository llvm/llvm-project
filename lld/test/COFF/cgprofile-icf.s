# REQUIRES: x86
# Test the compatibility of ICF and cgprofile.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t
# RUN: echo "A B 100" > %t.call_graph
# RUN: echo "A C 40" >> %t.call_graph
# RUN: echo "C D 61" >> %t.call_graph
# RUN: lld-link /subsystem:console /entry:A %t /call-graph-ordering-file:%t.call_graph /out:%t2 /debug:symtab /opt:icf
# RUN: llvm-nm --numeric-sort %t2 | FileCheck %s
# RUN: lld-link /subsystem:console /entry:A %t /call-graph-ordering-file:%t.call_graph /out:%t2 /debug:symtab
# RUN: llvm-nm --numeric-sort %t2 | FileCheck %s --check-prefix=NOICF

    .section    .text,"x",one_only,D
    .globl  D
D:
    mov $60, %rax
    retq

    .section    .text,"x",one_only,C
    .globl  C
C:
    mov $60, %rax
    retq

    .section    .text,"x",one_only,B
    .globl  B
B:
    mov $2, %rax
    retq

    .section    .text,"x",one_only,A
    .globl  A
A:
    mov $42, %rax
    retq

# CHECK: 140001000 T A
# CHECK: 140001008 T C
# CHECK: 140001008 T D
# CHECK: 140001010 T B

# NOICF: 140001000 T A
# NOICF: 140001008 T B
# NOICF: 140001010 T C
# NOICF: 140001018 T D
