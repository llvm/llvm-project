# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t --icf=all -shared
# RUN: llvm-readelf --sections --dyn-symbols %t | FileCheck %s

# We used to mark bar as absolute.

# CHECK: [ 5] .text
# CHECK: [[ADDR:[0-9a-f]+]] 0 NOTYPE  GLOBAL DEFAULT   5 foo
# CHECK: [[ADDR]]           0 NOTYPE  GLOBAL DEFAULT   5 bar

# The nop makes the test more interesting by making the offset of
# text.f non zero.

nop

        .section        .text.f,"ax",@progbits
        .globl  foo
foo:
        retq

        .section        .text.g,"ax",@progbits
        .globl  bar
bar:
        retq
