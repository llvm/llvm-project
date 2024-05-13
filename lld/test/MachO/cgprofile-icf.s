# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t

# RUN: %lld -e A %t -o %t.out --icf=all
# RUN: llvm-nm --numeric-sort %t.out | FileCheck %s
# RUN: %lld -e A %t -o %t2.out
# RUN: llvm-nm --numeric-sort %t2.out | FileCheck %s --check-prefix=NOICF

.text
    .globl  D
D:
    mov $60, %rax
    retq

    .globl  C
C:
    mov $60, %rax
    retq

    .globl  B
B:
    mov $2, %rax
    retq

    .globl  A
A:
    mov $42, %rax
    retq

.cg_profile A, B, 100
.cg_profile A, C,  40
.cg_profile C, D,  61

.subsections_via_symbols

# CHECK:      [[#%.16x,A:]]   T A
# CHECK-NEXT: [[#%.16x,A+8]]  T C
# CHECK-NEXT: [[#%.16x,A+8]]  T D
# CHECK-NEXT: [[#%.16x,A+16]] T B

# NOICF:      [[#%.16x,A:]]   T A
# NOICF-NEXT: [[#%.16x,A+8]]  T B
# NOICF-NEXT: [[#%.16x,A+16]] T C
# NOICF-NEXT: [[#%.16x,A+24]] T D
