# REQUIRES: x86

# Equal-sized IMAGE_COMDAT_SELECT_LARGEST candidates may be selected
# arbitrarily, but the leader and its associative section must always come from
# the same candidate. The output must never contain a mixed group.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %t.dir/a.s -o %t.a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %t.dir/b.s -o %t.b.obj

# RUN: lld-link /opt:ref /include:leader /dll /noentry /nodefaultlib \
# RUN:   %t.a.obj %t.b.obj /out:%t.ref.ab.exe
# RUN: llvm-objdump -s %t.ref.ab.exe | FileCheck %s

# RUN: lld-link /opt:noref /include:leader /dll /noentry /nodefaultlib \
# RUN:   %t.a.obj %t.b.obj /out:%t.noref.ab.exe
# RUN: llvm-objdump -s %t.noref.ab.exe | FileCheck %s

# RUN: lld-link /opt:ref /include:leader /dll /noentry /nodefaultlib \
# RUN:   %t.b.obj %t.a.obj /out:%t.ref.ba.exe
# RUN: llvm-objdump -s %t.ref.ba.exe | FileCheck %s

# RUN: lld-link /opt:noref /include:leader /dll /noentry /nodefaultlib \
# RUN:   %t.b.obj %t.a.obj /out:%t.noref.ba.exe
# RUN: llvm-objdump -s %t.noref.ba.exe | FileCheck %s

# CHECK: Contents of section .text:
# CHECK-NEXT: {{[0-9a-f]+}} {{(aaaaaaaa 11111111|bbbbbbbb 22222222)}}
# CHECK-NOT: aaaaaaaa 22222222
# CHECK-NOT: bbbbbbbb 11111111

#--- a.s
        .section .text$assoc, "xr", associative, leader
        .long 0xaaaaaaaa

        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111

#--- b.s
        .section .text$assoc, "xr", associative, leader
        .long 0xbbbbbbbb

        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x22222222
