# REQUIRES: x86

# A public symbol defined only in a superseded IMAGE_COMDAT_SELECT_LARGEST
# group must not be considered by MinGW automatic export. In particular, it
# must not be emitted with an RVA relative to a section absent from the image.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj

# Check both input orders. The small-first order exercises removal of an
# already prevailing definition; the large-first order exercises a candidate
# which never prevails.
# RUN: lld-link /lldmingw /dll /noentry /nodefaultlib /noimplib \
# RUN:   %t.small.obj %t.large.obj /out:%t.small-large.dll
# RUN: llvm-readobj --coff-exports %t.small-large.dll | \
# RUN:   FileCheck --implicit-check-not=only_small %s
# RUN: lld-link /lldmingw /dll /noentry /nodefaultlib /noimplib \
# RUN:   %t.large.obj %t.small.obj /out:%t.large-small.dll
# RUN: llvm-readobj --coff-exports %t.large-small.dll | \
# RUN:   FileCheck --implicit-check-not=only_small %s

# CHECK: Export {
# CHECK: Name: leader
# CHECK: RVA: 0x1000
# CHECK: }
# CHECK-NOT: Export {

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11

        .globl only_small
only_small:
        .byte 0x22

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
