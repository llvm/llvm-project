# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl foo; .hidden foo; foo = 42' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: ld.lld %t.o %t2.o -o %t.so -shared
# RUN: llvm-readelf -r -S -x .got -x .data %t.so | FileCheck %s

## Test we don't create relocations for non-preemptable absolute symbols.

# CHECK: There are no relocations in this file.
# CHECK: section '.got':
# CHECK: 0x000022b8 2a000000 00000000

## .got - (.data+8) = 0xfffff0ef
# CHECK: section '.data':
# CHECK: 0x000032c0 2a000000 00000000 f0efffff

.data
.quad foo
.long foo@gotpcrel
