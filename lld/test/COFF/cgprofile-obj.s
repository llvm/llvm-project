# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t
# RUN: lld-link /subsystem:console /entry:A %t /out:%t2 /debug:symtab
# RUN: llvm-nm --numeric-sort %t2 | FileCheck %s
# RUN: lld-link /call-graph-profile-sort:no /subsystem:console /entry:A %t /out:%t3 /debug:symtab
# RUN: llvm-nm --numeric-sort %t3 | FileCheck %s --check-prefix=NO-CG

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

# CHECK: 140001000 T A
# CHECK: 140001001 T B
# CHECK: 140001002 T C
# CHECK: 140001003 t D


# NO-CG: 140001000 t D
# NO-CG: 140001001 T C
# NO-CG: 140001002 T B
# NO-CG: 140001003 T A
