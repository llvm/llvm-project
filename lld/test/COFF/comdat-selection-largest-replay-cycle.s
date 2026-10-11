# REQUIRES: x86
#
# Build a ring in which each deferred COMDAT leader name is also published as
# a secondary definition by another replaceable group. This stresses the
# cycle/input-order fallback in deferred leader replay. The linker must
# terminate, drain all deferred providers, and keep each selected group
# internally consistent in either input order.
#
# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-small.s -o %t.root-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/a.s -o %t.a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/b.s -o %t.b.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/c.s -o %t.c.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-large.s -o %t.root-large.obj
#
# RUN: lld-link /force:multiple /dll /noentry /nodefaultlib /opt:noref \
# RUN:   /include:a /include:b /include:c %t.root-small.obj %t.a.obj \
# RUN:   %t.b.obj %t.c.obj %t.root-large.obj /out:%t.abc.dll
# RUN: llvm-objdump -s %t.abc.dll | FileCheck %s
#
# RUN: lld-link /force:multiple /dll /noentry /nodefaultlib /opt:noref \
# RUN:   /include:a /include:b /include:c %t.root-small.obj %t.c.obj \
# RUN:   %t.b.obj %t.a.obj %t.root-large.obj /out:%t.cba.dll
# RUN: llvm-objdump -s %t.cba.dll | FileCheck %s
#
# CHECK: Contents of section .data:
# CHECK-DAG: aaaaaaaa
# CHECK-DAG: bbbbbbbb
# CHECK-DAG: cccccccc
# CHECK-NOT: 11111111
# CHECK-NOT: 22222222
# CHECK-NOT: 33333333
#
#--- root-small.s
        .section .data$root, "dw", largest, root
        .globl root
root:
        .byte 0

        .globl a
a:
        .long 0x11111111
        .globl b
b:
        .long 0x22222222
        .globl c
c:
        .long 0x33333333
#
#--- a.s
        .section .data$a, "dw", largest, a
        .globl a
a:
        .space 16, 0xaa
        .globl b
b:
        .long 0xaaaaaaaa
#
#--- b.s
        .section .data$b, "dw", largest, b
        .globl b
b:
        .space 16, 0xbb
        .globl c
c:
        .long 0xbbbbbbbb
#
#--- c.s
        .section .data$c, "dw", largest, c
        .globl c
c:
        .space 16, 0xcc
        .globl a
a:
        .long 0xcccccccc
#
#--- root-large.s
        .section .data$root, "dw", largest, root
        .globl root
root:
        .space 64, 0x44
