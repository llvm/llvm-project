# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -e A %t/test.o -order_file %t/order_file -o %t/test 
# RUN: llvm-nm --numeric-sort %t/test | FileCheck %s
# RUN: %lld -e A %t/test.o -o %t/test 
# RUN: llvm-nm --numeric-sort %t/test | FileCheck %s --check-prefix NO-ORDER


#--- order_file
B
A

#--- test.s

.text
    .globl  D
D:
    retq

    .globl  C
C:
    retq

    .globl  B
B:
    retq

    .globl  A
A:
    retq

.cg_profile A, B, 100
.cg_profile A, C,  40
.cg_profile C, D,  61

.subsections_via_symbols

# CHECK:      T B
# CHECK-NEXT: T A
# CHECK-NEXT: T C
# CHECK-NEXT: T D

# NO-ORDER:      T A
# NO-ORDER-NEXT: T B
# NO-ORDER-NEXT: T C
# NO-ORDER-NEXT: T D

