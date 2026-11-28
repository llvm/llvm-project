# REQUIRES: x86

## Two static functions named f in different files must each get their own
## place in the call graph order.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %t/a.s -o %t/a.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %t/b.s -o %t/b.obj
# RUN: lld-link /subsystem:console /entry:main %t/a.obj %t/b.obj /out:%t/out.exe /debug:symtab
# RUN: llvm-nm --numeric-sort %t/out.exe | FileCheck %s

## The two f have different sizes, so the addresses show which one is where.
# CHECK:      140001000 T main
# CHECK-NEXT: 140001001 t f
# CHECK-NEXT: 140001003 T g
# CHECK-NEXT: 140001004 t f

#--- a.s
    .section .text,"ax",one_only,main
    .globl main
main:
    retq

    .section .text,"ax",one_only,f
f:
    nop
    retq

    .cg_profile main, f, 100

#--- b.s
    .section .text,"ax",one_only,g
    .globl g
g:
    retq

    .section .text,"ax",one_only,f
f:
    nop
    nop
    retq

    .cg_profile g, f, 10
