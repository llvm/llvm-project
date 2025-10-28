# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld -e A a.o --symbol-ordering-file=order --call-graph-profile-sort=hfsort -o out
# RUN: llvm-nm --numeric-sort out | FileCheck %s
# RUN: ld.lld -e A a.o --call-graph-profile-sort=hfsort -o out1
# RUN: llvm-nm --numeric-sort out1 | FileCheck %s --check-prefix=ONLY-CG

#--- order
B
A

#--- a.s
.section .text.D,"ax"; .globl D; D:
  retq

.section .text.C,"ax"; .globl C; C:
  call D

.section .text.B,"ax"; .globl B; B:
  retq

.section .text.A,"ax"; .globl A; A:
  call B
  call C

.cg_profile A, B, 100
.cg_profile A, C,  40
.cg_profile C, D,  61

# CHECK:      T B
# CHECK-NEXT: T A
# CHECK-NEXT: T C
# CHECK-NEXT: T D

# ONLY-CG:      T A
# ONLY-CG-NEXT: T B
# ONLY-CG-NEXT: T C
# ONLY-CG-NEXT: T D
