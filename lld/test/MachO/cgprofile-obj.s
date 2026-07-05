# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem -e A -o %t.out %t.o
# RUN: llvm-nm --numeric-sort %t.out | FileCheck %s
# RUN: %lld --no-call-graph-profile-sort -lSystem -e A -o %t.out %t.o
# RUN: llvm-nm --numeric-sort %t.out | FileCheck %s --check-prefix=NO-CG

.text

D:
  retq
.globl C
C:
  retq
.globl B
B:
  retq
.globl A
A:
Aa:
  retq
.subsections_via_symbols



    .cg_profile A, B, 10
    .cg_profile A, B, 10
    .cg_profile Aa, B, 80
    .cg_profile A, C, 40
    .cg_profile B, C, 30
    .cg_profile C, D, 90

# CHECK: 00000001000002c8 T A
# CHECK: 00000001000002c9 T B
# CHECK: 00000001000002ca T C
# CHECK: 00000001000002cb t D

# NO-CG: 00000001000002c8 t D
# NO-CG: 00000001000002c9 T C
# NO-CG: 00000001000002ca T B
# NO-CG: 00000001000002cb T A
