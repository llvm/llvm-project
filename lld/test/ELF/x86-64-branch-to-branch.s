# REQUIRES: x86

## Test that the branch-to-branch optimization follows the links
## from f1 -> f2 -> f3 and updates all references to point to f3.
 
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t --branch-to-branch
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,B2B %s
# RUN: ld.lld %t.o -o %t -O2
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,B2B %s

## Test that branch-to-branch is disabled by default.

# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,NOB2B %s
# RUN: ld.lld %t.o -o %t -O2 --no-branch-to-branch
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,NOB2B %s

## Test that branch-to-branch is disabled for preemptible symbols.

# RUN: ld.lld %t.o -o %t --branch-to-branch -shared
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,NOB2B %s

.section .rodata.vtable,"a"
.globl vtable
vtable:
# B2B: Contents of section .rodata:
# B2B-NEXT: [[VF:[0-9a-f]{8}]]
.4byte f1@PLT - vtable
# B2B-SAME: [[VF]]
.4byte f2@PLT - vtable
# B2B-SAME: [[VF]]
.4byte f3@PLT - vtable

.section .text._start,"ax"
.globl _start
# CHECK: <_start>:
_start:
# B2B-NEXT: jmp {{.*}} <f3>
# NOB2B-NEXT: jmp {{.*}} <f1{{.*}}>
jmp f1
# B2B-NEXT: jmp {{.*}} <f3>
# NOB2B-NEXT: jmp {{.*}} <f2{{.*}}>
jmp f2

.section .text.f1,"ax"
.globl f1
f1:
jmp f2

.section .text.f2,"ax"
.globl f2
# CHECK: <f2>:
f2:
# CHECK-NEXT: jmp {{.*}} <f3{{.*}}>
jmp f3

.section .text.f3,"ax"
.globl f3
f3:
ret
