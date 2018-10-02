# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld -e A %t -o %t2
# RUN: llvm-readobj -symbols %t2 | FileCheck %s

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

# CHECK:          Name: D
# CHECK-NEXT:     Value: 0x201003
# CHECK:          Name: A
# CHECK-NEXT:     Value: 0x201000
# CHECK:          Name: B
# CHECK-NEXT:     Value: 0x201001
# CHECK:          Name: C
# CHECK-NEXT:     Value: 0x201002
