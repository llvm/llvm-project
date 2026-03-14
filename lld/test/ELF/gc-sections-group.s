# REQUIRES: x86
## Check that group members are retained or discarded as a unit, and
## non-SHF_ALLOC sections in a group are subject to garbage collection,
## if at least one member has the SHF_ALLOC flag.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gc-sections %t.o -o %t.dead
# RUN: llvm-readobj -S %t.dead | FileCheck %s --check-prefix=CHECK-DEAD

## .mynote.ccc is retained because it is not in a group.
# CHECK-DEAD-NOT: Name: .myanote.aaa
# CHECK-DEAD-NOT: Name: .mytext.aaa
# CHECK-DEAD-NOT: Name: .mybss.aaa
# CHECK-DEAD-NOT: Name: .mynote.aaa
# CHECK-DEAD-NOT: Name: .myanote.bbb
# CHECK-DEAD-NOT: Name: .mytext.bbb
# CHECK-DEAD-NOT: Name: .mybss.bbb
# CHECK-DEAD-NOT: Name: .mynote.bbb
# CHECK-DEAD:     Name: .mynote.ccc

# RUN: ld.lld --gc-sections %t.o -o %t -e anote_aaa
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-GROUP
# RUN: ld.lld --gc-sections %t.o -o %t -e aaa
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-GROUP
# RUN: ld.lld --gc-sections %t.o -o %t -e bss_aaa
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-GROUP

## note_zero as the entry point does not make much sense because it is defined
## in a non-SHF_ALLOC section. This is just to demonstrate the behavior.
# RUN: ld.lld --gc-sections %t.o -o %t -e note_aaa
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-GROUP

# CHECK-LIVE-GROUP: Name: .myanote.aaa
# CHECK-LIVE-GROUP: Name: .mytext.aaa
# CHECK-LIVE-GROUP: Name: .mybss.aaa
# CHECK-LIVE-GROUP: Name: .mynote.aaa
# CHECK-LIVE-GROUP-NOT: Name: .myanote.bbb
# CHECK-LIVE-GROUP-NOT: Name: .mytext.bbb
# CHECK-LIVE-GROUP-NOT: Name: .mybss.bbb
# CHECK-LIVE-GROUP-NOT: Name: .mynote.bbb
# CHECK-LIVE-GROUP: Name: .mynote.ccc

# RUN: ld.lld --gc-sections %t.o -o %t -e anote_bbb
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-COMDAT
# RUN: ld.lld --gc-sections %t.o -o %t -e bbb
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-COMDAT
# RUN: ld.lld --gc-sections %t.o -o %t -e bss_bbb
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-COMDAT

## note_bbb as the entry point does not make much sense because it is defined
## in a non-SHF_ALLOC section. This is just to demonstrate the behavior.
# RUN: ld.lld --gc-sections %t.o -o %t -e note_bbb
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE-COMDAT

# CHECK-LIVE-COMDAT-NOT: Name: .myanote.aaa
# CHECK-LIVE-COMDAT-NOT: Name: .mytext.aaa
# CHECK-LIVE-COMDAT-NOT: Name: .mybss.aaa
# CHECK-LIVE-COMDAT-NOT: Name: .mynote.aaa
# CHECK-LIVE-COMDAT: Name: .myanote.bbb
# CHECK-LIVE-COMDAT: Name: .mytext.bbb
# CHECK-LIVE-COMDAT: Name: .mybss.bbb
# CHECK-LIVE-COMDAT: Name: .mynote.bbb
# CHECK-LIVE-COMDAT: Name: .mynote.ccc

## These sections are in a zero flag group `aaa`.
.globl anote_aaa, aaa, bss_aaa, note_aaa

.section .myanote.aaa,"aG",@note,aaa
anote_aaa:
.byte 0

.section .mytext.aaa,"axG",@progbits,aaa
aaa:
.byte 0

.section .mybss.aaa,"awG",@nobits,aaa
bss_aaa:
.byte 0

.section .mynote.aaa,"G",@note,aaa
note_aaa:
.byte 0

## These sections are in a COMDAT group `bbb`.
.globl anote_bbb, bbb, bss_bbb, note_bbb

.section .myanote.bbb,"aG",@note,bbb,comdat
anote_bbb:
.byte 0

.section .mytext.bbb,"axG",@progbits,bbb,comdat
bbb:
.byte 0

.section .mybss.bbb,"awG",@nobits,bbb,comdat
bss_bbb:
.byte 0

.section .mynote.bbb,"G",@note,bbb,comdat
note_bbb:
.byte 0

## This section isn't in any group.
.section .mynote.ccc,"",@note
.byte 0
