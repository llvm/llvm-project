# REQUIRES: x86

# Test symbols other than the COMDAT leader when a larger definition replaces
# an earlier prevailing IMAGE_COMDAT_SELECT_LARGEST group.
#
# The positive cases define the same secondary external symbol in both
# candidates. The prevailing secondary symbol, local symbol, and associative
# relocations must all refer to the final largest group.
#
# The negative cases define only_small exclusively in the smaller candidate.
# Once that candidate is superseded, only_small must no longer provide a
# definition, independently of input order.

# RUN: split-file %s %t.dir

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/only-small.s -o %t.only-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/only-large.s -o %t.only-large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/only-root.s -o %t.only-root.obj

# Check replacement when the smaller candidate is seen first.

# RUN: lld-link /opt:noref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.noref.small-large.exe
# RUN: llvm-objdump -s %t.noref.small-large.exe | FileCheck %s

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.ref.small-large.exe
# RUN: llvm-objdump -s %t.ref.small-large.exe | FileCheck %s

# Check that a smaller candidate seen later does not replace the prevailing
# larger candidate.

# RUN: lld-link /opt:noref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.large.obj %t.small.obj %t.root.obj \
# RUN:   /out:%t.noref.large-small.exe
# RUN: llvm-objdump -s %t.noref.large-small.exe | FileCheck %s

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.large.obj %t.small.obj %t.root.obj \
# RUN:   /out:%t.ref.large-small.exe
# RUN: llvm-objdump -s %t.ref.large-small.exe | FileCheck %s

# A symbol defined only by the superseded smaller group must no longer
# provide a definition after the larger group prevails.
#
# Cover all six permutations of the smaller candidate, larger candidate,
# and the object containing the undefined reference.

# Smaller candidate, larger candidate, reference.

# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.only-small.obj %t.only-large.obj %t.only-root.obj \
# RUN:   /out:%t.only-small-large-root.exe 2>&1 | \
# RUN:   FileCheck --check-prefix=ONLY-SMALL %s

# Smaller candidate, reference, larger candidate.

# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.only-small.obj %t.only-root.obj %t.only-large.obj \
# RUN:   /out:%t.only-small-root-large.exe 2>&1 | \
# RUN:   FileCheck --check-prefix=ONLY-SMALL %s

# Larger candidate, smaller candidate, reference.

# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.only-large.obj %t.only-small.obj %t.only-root.obj \
# RUN:   /out:%t.only-large-small-root.exe 2>&1 | \
# RUN:   FileCheck --check-prefix=ONLY-SMALL %s

# Larger candidate, reference, smaller candidate.

# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.only-large.obj %t.only-root.obj %t.only-small.obj \
# RUN:   /out:%t.only-large-root-small.exe 2>&1 | \
# RUN:   FileCheck --check-prefix=ONLY-SMALL %s

# Reference, smaller candidate, larger candidate.

# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.only-root.obj %t.only-small.obj %t.only-large.obj \
# RUN:   /out:%t.only-root-small-large.exe 2>&1 | \
# RUN:   FileCheck --check-prefix=ONLY-SMALL %s

# Reference, larger candidate, smaller candidate.

# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.only-root.obj %t.only-large.obj %t.only-small.obj \
# RUN:   /out:%t.only-root-large-small.exe 2>&1 | \
# RUN:   FileCheck --check-prefix=ONLY-SMALL %s

# CHECK: Contents of section .text:
# CHECK-NEXT:  140001000 44444444 55555555 66666666 c3
# CHECK: Contents of section .rdata:
# The first three RVAs come from the associative section. The final two come
# from the root section. They must all point into the prevailing large chunk:
# leader = 0x1000, secondary = 0x1004, local = 0x1008.
# CHECK-NEXT:  140002000 00100000 04100000 08100000 00100000
# CHECK-NEXT:  140002010 04100000

# ONLY-SMALL: error: undefined symbol: only_small
# ONLY-SMALL-NOT: duplicate symbol: only_small

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11

        .globl secondary
secondary:
        .byte 0x22

local:
        .byte 0x33

        .section .rdata$assoc, "dr", associative, leader
        .rva leader
        .rva secondary
        .rva local

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444

        .globl secondary
secondary:
        .long 0x55555555

local:
        .long 0x66666666

        .section .rdata$assoc, "dr", associative, leader
        .rva leader
        .rva secondary
        .rva local

#--- root.s
        .section .text$root, "xr"
        .globl entry
entry:
        retq

        .section .rdata$refs, "dr"
        .rva leader
        .rva secondary

#--- only-small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11

        .globl only_small
only_small:
        .byte 0x22

#--- only-large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444

#--- only-root.s
        .section .text$root, "xr"
        .globl entry
entry:
        leaq only_small(%rip), %rax
        retq
