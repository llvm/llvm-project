# REQUIRES: x86

## The call graph sort only moves sections with a leader symbol. Plain .text
## sections keep their input order.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %t/a.s -o %t/a.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %t/b.s -o %t/b.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %t/c.s -o %t/c.obj
# RUN: lld-link /subsystem:console /entry:main %t/a.obj %t/b.obj %t/c.obj /out:%t/out.exe /debug:symtab
# RUN: llvm-nm --numeric-sort %t/out.exe | FileCheck %s

## main calls h, but all three stay in input order.
# CHECK:      140001000 T main
# CHECK-NEXT: 140001004 T g
# CHECK-NEXT: 140001008 T h

#--- a.s
    .text
    .globl main
main:
    retq

    .cg_profile main, h, 100

#--- b.s
    .text
    .globl g
g:
    nop
    nop
    retq

#--- c.s
    .text
    .globl h
h:
    nop
    retq
