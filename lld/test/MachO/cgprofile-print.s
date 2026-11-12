# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t
# RUN: %lld -e A %t -o %t2  --print-symbol-order=%t3
# RUN: FileCheck %s --input-file %t3

# CHECK: B
# CHECK-NEXT: C
# CHECK-NEXT: D
# CHECK-NEXT: A

.text
.globl  A
A:
 nop

.globl  B
B:
 nop

.globl  C
C:
 nop

.globl  D
D:
 nop

.subsections_via_symbols

.cg_profile A, B, 5
.cg_profile B, C, 50
.cg_profile C, D, 40
.cg_profile D, B, 10
