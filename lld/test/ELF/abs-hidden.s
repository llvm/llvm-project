# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl foo; .hidden foo; foo = 42' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: ld.lld %t.o %t2.o -o %t.so -shared
# RUN: llvm-readelf -r -S -x .got -x .data %t.so | FileCheck %s

## Test we don't create relocations for non-preemptable absolute symbols.

# CHECK: There are no relocations in this file.
# CHECK: section '.got':
# CHECK: 0x00001070 2a000000 00000000

## .got - (.data+8) = 0xfffff068
# CHECK: section '.data':
# CHECK: 0x00002000 2a000000 00000000 68f0ffff

.data
.quad foo
.long foo@gotpcrel
