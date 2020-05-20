# REQUIRES: x86

# RUN: echo -e "EXPORTS\n  foo.bar" > %t.def

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -defsym drectve=0 %s -o %t.obj
# RUN: lld-link -entry:dllmain -dll -def:%t.def %t.obj -out:%t.1.dll
# RUN: llvm-readobj --coff-exports %t.1.dll | FileCheck %s

# RUN: lld-link -entry:dllmain -dll %t.obj -out:%t.2.dll -export:foo.bar
# RUN: llvm-readobj --coff-exports %t.2.dll | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -defsym drectve=1 %s -o %t.drectve.obj
# RUN: lld-link -entry:dllmain -dll %t.drectve.obj -out:%t.3.dll
# RUN: llvm-readobj --coff-exports %t.3.dll | FileCheck %s

# CHECK: Name: foo.bar

        .text
        .globl  dllmain
        .globl  foo.bar
dllmain:
        ret
foo.bar:
        ret

.if drectve==1
        .section .drectve
        .ascii "-export:foo.bar"
.endif
