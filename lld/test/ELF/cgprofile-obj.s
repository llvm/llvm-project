# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld -e A %t.o -o %t
# RUN: llvm-nm --no-sort %t | FileCheck %s --check-prefix=CG-OBJ
# RUN: ld.lld --call-graph-profile-sort=none -e A %t.o -o %t
# RUN: llvm-nm --no-sort %t | FileCheck %s --check-prefix=NO-CG
## --no-call-graph-profile-sort is an alias for --call-graph-profile-sort=none.
# RUN: ld.lld --no-call-graph-profile-sort -e A %t.o -o %t1
# RUN: cmp %t %t1
# RUN: echo "D A 200" > %t.call_graph
# RUN: ld.lld -e A %t.o -call-graph-ordering-file=%t.call_graph -o %t2
# RUN: llvm-nm --no-sort %t2 | FileCheck %s --check-prefix=CG-OBJ-OF

    .section    .text.D,"ax",@progbits
D:
    retq

    .section    .text.C,"ax",@progbits
    .globl  C
C:
    retq

    .section    .text.B,"ax",@progbits
    .globl  B
B:
    retq

    .section    .text.A,"ax",@progbits
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

# CG-OBJ:      0000000000201123 t D
# CG-OBJ-NEXT: 0000000000201120 t Aa
# CG-OBJ-NEXT: 0000000000201122 T C
# CG-OBJ-NEXT: 0000000000201121 T B
# CG-OBJ-NEXT: 0000000000201120 T A

# NO-CG:      0000000000201120 t D
# NO-CG-NEXT: 0000000000201123 t Aa
# NO-CG-NEXT: 0000000000201121 T C
# NO-CG-NEXT: 0000000000201122 T B
# NO-CG-NEXT: 0000000000201123 T A

# CG-OBJ-OF:      0000000000201120 t D
# CG-OBJ-OF-NEXT: 0000000000201121 t Aa
# CG-OBJ-OF-NEXT: 0000000000201124 T C
# CG-OBJ-OF-NEXT: 0000000000201125 T B
# CG-OBJ-OF-NEXT: 0000000000201121 T A
