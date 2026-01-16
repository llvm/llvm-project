# REQUIRES: aarch64

## Test that the branch-to-branch optimization follows the links
## from f1 -> f2 -> f3 and updates all references to point to f3.

# RUN: llvm-mc -filetype=obj -triple=aarch64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t --branch-to-branch --emit-relocs
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,B2B %s
# RUN: llvm-objdump -r %t | FileCheck --check-prefixes=RELOC,B2B-RELOC %s
# RUN: ld.lld %t.o -o %t -O2 --emit-relocs
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,B2B %s
# RUN: llvm-objdump -r %t | FileCheck --check-prefixes=RELOC,B2B-RELOC %s

## Test that branch-to-branch is disabled by default.

# RUN: ld.lld %t.o -o %t --emit-relocs
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,NOB2B %s
# RUN: llvm-objdump -r %t | FileCheck --check-prefixes=RELOC,NOB2B-RELOC %s
# RUN: ld.lld %t.o -o %t -O2 --no-branch-to-branch --emit-relocs
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,NOB2B %s
# RUN: llvm-objdump -r %t | FileCheck --check-prefixes=RELOC,NOB2B-RELOC %s

## Test that branch-to-branch is disabled for preemptible symbols.

# RUN: ld.lld %t.o -o %t --branch-to-branch -shared --emit-relocs
# RUN: llvm-objdump -d -s %t | FileCheck --check-prefixes=CHECK,NOB2B %s
# RUN: llvm-objdump -r %t | FileCheck --check-prefixes=RELOC,NOB2B-RELOC %s

.section .rodata.vtable,"a"
.globl vtable
vtable:
# B2B: Contents of section .rodata:
# RELOC: RELOCATION RECORDS FOR [.rodata]:
# RELOC-NEXT: OFFSET
# B2B-NEXT: [[VF:[0-9a-f]{8}]]
# B2B-RELOC-NEXT: R_AARCH64_PLT32 f3
# NOB2B-RELOC-NEXT: R_AARCH64_PLT32 f1
.4byte f1@PLT - vtable
# B2B-SAME: [[VF]]
# B2B-RELOC-NEXT: R_AARCH64_PLT32 f3+0x4
# NOB2B-RELOC-NEXT: R_AARCH64_PLT32 f2+0x4
.4byte f2@PLT - vtable
# B2B-SAME: [[VF]]
# RELOC-NEXT: R_AARCH64_PLT32 f3+0x8
.4byte f3@PLT - vtable

.section .text._start,"ax"
.globl _start
# CHECK: <_start>:
# RELOC: RELOCATION RECORDS FOR [.text]:
# RELOC-NEXT: OFFSET
_start:
.cfi_startproc
# B2B: bl {{.*}} <f3>
# B2B-RELOC-NEXT: R_AARCH64_CALL26 f3
# NOB2B: bl {{.*}} <f1{{.*}}>
# NOB2B-RELOC-NEXT: R_AARCH64_CALL26 f1
bl f1
# B2B: b {{.*}} <f3>
# B2B-RELOC-NEXT: R_AARCH64_JUMP26 f3
# NOB2B: b {{.*}} <f2{{.*}}>
# NOB2B-RELOC-NEXT: R_AARCH64_JUMP26 f2
b f2
.cfi_endproc

.section .text.f1,"ax"
.globl f1
f1:
# B2B-RELOC-NEXT: R_AARCH64_JUMP26 f3
# NOB2B-RELOC-NEXT: R_AARCH64_JUMP26 f2
b f2

.section .text.f2,"ax"
.globl f2
# CHECK: <f2>:
f2:
# CHECK-NEXT: b {{.*}} <f3{{.*}}>
# RELOC-NEXT: R_AARCH64_JUMP26 f3
b f3

.section .text.f3,"ax"
.globl f3
f3:
ret
