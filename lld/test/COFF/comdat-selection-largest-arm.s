# REQUIRES: arm

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple thumbv7-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple thumbv7-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: lld-link /dll /noentry /nodefaultlib /include:leader \
# RUN:   %t.small.obj %t.large.obj /out:%t.dll
# RUN: llvm-objdump -s %t.dll | FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# LARGE: Contents of section .text:
# LARGE: 44444444
# LARGE: Contents of section .rdata:
# LARGE: 55555555 bbbbbbbb

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111
        .section .rdata$assoc, "dr", associative, leader
        .long 0xaaaaaaaa

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
        .space 28, 0x44
        .section .rdata$assoc, "dr", associative, leader
        .long 0x55555555
        .long 0xbbbbbbbb
