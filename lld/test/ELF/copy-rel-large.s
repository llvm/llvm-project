# REQUIRES: x86

## Test symbols larger than 2**32 can be copy relocated.

# RUN: echo '.globl foo; .type foo,@object; foo: .byte 0; .size foo, 0x100000001' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared %t1.o -o %t1.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readelf -S --dyn-syms %t | FileCheck %s

# CHECK: [ 8] .bss.rel.ro
# CHECK: 4294967297 OBJECT GLOBAL DEFAULT  8 foo

        .global _start
_start:
        .quad foo
