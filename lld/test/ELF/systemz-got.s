# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname=%t2.so -o %t2.so

# RUN: ld.lld -dynamic-linker /lib/ld64.so.1 %t.o %t2.so -o %t
# RUN: llvm-readelf -S -r  %t | FileCheck %s

# CHECK: .got              PROGBITS        {{.*}} {{.*}} 000020 00  WA  0   0  8

# CHECK: Relocation section '.rela.dyn' at offset {{.*}} contains 1 entries:
# CHECK: {{.*}}  000000010000000a R_390_GLOB_DAT         0000000000000000 bar + 0

.global _start
_start:
	lgrl  %r1,bar@GOT
