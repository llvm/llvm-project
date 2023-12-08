# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/1.s -o %t/1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/2.s -o %t/2.o
# RUN: ld.lld -shared -soname=t2 %t/2.o -o %t/2.so
# RUN: ld.lld %t/1.o %t/2.so -o %t/1
# RUN: llvm-readelf -S -r --dyn-syms %t/1 | FileCheck %s
# RUN: ld.lld --gc-sections -pie %t/1.o %t/2.so -o %t/1
# RUN: llvm-readelf -S -r --dyn-syms %t/1 | FileCheck %s

# CHECK: [[#BSS:]]] .bss

# CHECK:      R_X86_64_COPY     [[#%x,]] a1 + 0
# CHECK-NEXT: R_X86_64_GLOB_DAT [[#%x,]] b1 + 0
# CHECK-NEXT: R_X86_64_COPY     [[#%x,]] b1 + 0
# CHECK-NEXT: R_X86_64_GLOB_DAT [[#%x,]] a2 + 0

# CHECK:         Value        Size Type    Bind   Vis     Ndx      Name
# CHECK:      [[#%x,ADDR:]]      1 OBJECT  GLOBAL DEFAULT [[#BSS]] a1
# CHECK-NEXT: {{0*}}[[#ADDR+1]]  1 OBJECT  WEAK   DEFAULT [[#BSS]] b1
# CHECK-NEXT: {{0*}}[[#ADDR+1]]  1 OBJECT  GLOBAL DEFAULT [[#BSS]] b2
# CHECK-NEXT: {{0*}}[[#ADDR]]    1 OBJECT  WEAK   DEFAULT [[#BSS]] a2
# CHECK-NEXT: {{0*}}[[#ADDR+1]]  1 OBJECT  GLOBAL DEFAULT [[#BSS]] b3

#--- 1.s
.global _start
_start:
movl $5, a1
mov $b1 - ., %eax
mov $b2 - ., %eax

## Test that a copy relocated alias may have GOT entry.
.weak a2, b1
movq a2@gotpcrel(%rip), %rax
movq b1@gotpcrel(%rip), %rcx

#--- 2.s
.data

.globl a1, b3
.weak a2, b1, b2

.type a1, @object
.type a2, @object
a1:
a2:
.byte 1
.size a1, 1
.size a2, 1

.type b1, @object
.type b2, @object
.type b3, @object
b1:
b2:
b3:
.byte 2
.size b1, 1
.size b2, 1
.size b3, 1
