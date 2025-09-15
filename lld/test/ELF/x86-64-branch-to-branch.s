# REQUIRES: x86

## Test that the branch-to-branch optimization follows the links
## from f1 -> f2 -> f3 and updates all references to point to f3.
 
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
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
# B2B-RELOC-NEXT: R_X86_64_PLT32 f3
# NOB2B-RELOC-NEXT: R_X86_64_PLT32 f1
.4byte f1@PLT - vtable
# B2B-SAME: [[VF]]
# B2B-RELOC-NEXT: R_X86_64_PLT32 f3+0x4
# NOB2B-RELOC-NEXT: R_X86_64_PLT32 f2+0x4
.4byte f2@PLT - vtable
# B2B-SAME: [[VF]]
# RELOC-NEXT: R_X86_64_PLT32 f3+0x8
.4byte f3@PLT - vtable

# For .rodata.f6
# B2B-RELOC-NEXT: R_X86_64_PLT32 f3-0x4
 
.section .text._start,"ax"
.globl _start
# CHECK: <_start>:
# RELOC: RELOCATION RECORDS FOR [.text]:
# RELOC-NEXT: OFFSET
_start:
.cfi_startproc
# B2B-NEXT: jmp {{.*}} <f3>
# B2B-RELOC-NEXT: R_X86_64_PLT32 f3-0x4
# NOB2B-NEXT: jmp {{.*}} <f1{{.*}}>
# NOB2B-RELOC-NEXT: R_X86_64_PLT32 f1-0x4
jmp f1
# B2B-NEXT: jmp {{.*}} <f3>
# B2B-RELOC-NEXT: R_X86_64_PLT32 f3-0x4
# NOB2B-NEXT: jmp {{.*}} <f2{{.*}}>
# NOB2B-RELOC-NEXT: R_X86_64_PLT32 f2-0x4
jmp f2
# This will assemble to a relocation pointing to an STT_SECTION for .text.f4
# with an addend, which looks similar to the relative vtable cases above but
# requires different handling of the addend so that we don't think this is
# branching to the `jmp f3` at the start of the target section.
# CHECK-NEXT: jmp {{.*}} <f4{{.*}}>
# RELOC-NEXT: R_X86_64_PLT32 .text+0x2e
jmp f4
# B2B-NEXT: jmp 0x[[IPLT:[0-9a-f]*]]
# RELOC-NEXT: R_X86_64_PLT32 f5-0x4
jmp f5
# B2B-NEXT: jmp {{.*}} <f6>
# RELOC-NEXT: R_X86_64_PLT32 f6-0x4
jmp f6
# B2B-NEXT: jmp {{.*}} <f7>
# RELOC-NEXT: R_X86_64_PLT32 f7-0x4
jmp f7
.cfi_endproc

.section .text.f1,"ax"
.globl f1
f1:
# B2B-RELOC-NEXT: R_X86_64_PLT32 f3-0x4
# NOB2B-RELOC-NEXT: R_X86_64_PLT32 f2-0x4
jmp f2

.section .text.f2,"ax"
.globl f2
# CHECK: <f2>:
f2:
# CHECK-NEXT: jmp {{.*}} <f3{{.*}}>
# RELOC-NEXT: R_X86_64_PLT32 f3-0x4
jmp f3

.section .text.f3,"ax"
.globl f3
f3:
# Test that a self-branch doesn't trigger an infinite loop.
# RELOC-NEXT: R_X86_64_PLT32 f3-0x4
jmp f3

.section .text.f4,"ax"
jmp f3
f4:
ret

.section .text.f5,"ax"
.type f5, @gnu_indirect_function
.globl f5
f5:
# RELOC-NEXT: R_X86_64_PLT32 f3-0x4
jmp f3

.section .rodata.f6,"a"
.globl f6
f6:
# RELOC-NEXT: R_X86_64_PLT32 f3-0x4
jmp f3

# RELOC: RELOCATION RECORDS FOR [.wtext.f7]:
# RELOC-NEXT: OFFSET

.section .wtext.f7,"awx"
.globl f7
f7:
# RELOC-NEXT: R_X86_64_PLT32 f3-0x4
jmp f3

# B2B: <.iplt>:
# B2B-NEXT: [[IPLT]]:
