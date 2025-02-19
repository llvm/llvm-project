# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t
# RUN: lld-link /subsystem:console /entry:A %t /out:%t2 /debug:symtab
# RUN: llvm-nm --numeric-sort %t2 | FileCheck %s --check-prefix=CG-OBJ
# RUN: lld-link /call-graph-profile-sort:no /subsystem:console /entry:A %t /out:%t3 /debug:symtab
# RUN: llvm-nm --numeric-sort %t3 | FileCheck %s --check-prefix=NO-CG
# RUN: echo "D A 200" > %t.call_graph
# RUN: lld-link /subsystem:console /entry:A %t /out:%t4 /debug:symtab /call-graph-ordering-file:%t.call_graph
# RUN: llvm-nm --numeric-sort %t4 | FileCheck %s --check-prefix=CG-OBJ-OF

    .section    .text,"ax", one_only, D
D:
 retq

    .section    .text,"ax", one_only, C
    .globl  C
C:
 retq

    .section    .text,"ax", one_only, B
    .globl  B
B:
 retq

    .section    .text,"ax", one_only, A
    .globl  A
A:
Aa:
 retq

    .cg_profile A, B, 10
    .cg_profile A, B, 10
    .cg_profile Aa, B, 80
    .cg_profile A, C, 40
    .cg_profile B, C, 30
    .cg_profile C, D, 90

# CG-OBJ:      140001000 T A
# CG-OBJ-NEXT: 140001000 t Aa
# CG-OBJ-NEXT: 140001001 T B
# CG-OBJ-NEXT: 140001002 T C
# CG-OBJ-NEXT: 140001003 t D

# NO-CG:      140001000 t D
# NO-CG-NEXT: 140001001 T C
# NO-CG-NEXT: 140001002 T B
# NO-CG-NEXT: 140001003 T A
# NO-CG-NEXT: 140001003 t Aa

# CG-OBJ-OF:      140001000 t D
# CG-OBJ-OF-NEXT: 140001001 T A
# CG-OBJ-OF-NEXT: 140001001 t Aa
# CG-OBJ-OF-NEXT: 140001004 T C
# CG-OBJ-OF-NEXT: 140001005 T B
