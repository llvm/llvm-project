# REQUIRES: x86

# A reference to the COMDAT leader must resolve to the final largest candidate.
# A superseded group must not be resurrected by section GC or interact
# incorrectly with ICF.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj /out:%t.ref.exe
# RUN: llvm-objdump -s %t.ref.exe | FileCheck %s

# RUN: lld-link /opt:noref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj /out:%t.noref.exe
# RUN: llvm-objdump -s %t.noref.exe | FileCheck %s

# RUN: lld-link /opt:ref /opt:icf /entry:entry /subsystem:console \
# RUN:   /nodefaultlib %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.ref-icf.exe
# RUN: llvm-objdump -s %t.ref-icf.exe | FileCheck %s

# RUN: lld-link /opt:noref /opt:icf /entry:entry /subsystem:console \
# RUN:   /nodefaultlib %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.noref-icf.exe
# RUN: llvm-objdump -s %t.noref-icf.exe | FileCheck %s

# CHECK: Contents of section .text:
# CHECK-NEXT:  140001000 bbbbbbbb 44444444
# CHECK-NOT: aaaaaaaa
# CHECK-NOT: 11111111

#--- small.s
        .section .text$assoc, "xr", associative, leader
        .long 0xaaaaaaaa

        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11

#--- large.s
        .section .text$assoc, "xr", associative, leader
        .long 0xbbbbbbbb

        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444

#--- root.s
        .section .text$root, "xr"
        .globl entry
entry:
        leaq leader(%rip), %rax
        retq
