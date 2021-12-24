# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/protected-shared.s -o %t2.o
# RUN: ld.lld -shared --soname=t2 %t2.o -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-readelf -s %t | FileCheck %s
# RUN: ld.lld %t2.so %t.o -o %t1
# RUN: llvm-readelf -s %t1 | FileCheck %s

# CHECK: Symbol table '.dynsym'
# CHECK:    Num:    Value          Size Type    Bind   Vis      Ndx Name
# CHECK:      1: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND foo

# CHECK: Symbol table '.symtab'
# CHECK:    Num:    Value          Size Type    Bind   Vis      Ndx Name
# CHECK:         0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND foo
# CHECK:         {{.*}}               0 NOTYPE  GLOBAL DEFAULT [[#]] bar

        .global  _start
_start:

        .data
        .quad foo

        .global bar
bar:
