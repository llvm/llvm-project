# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t --icf=all -shared
# RUN: llvm-readelf --dyn-symbols %t | FileCheck %s

# We used to mark bar as absolute.

# CHECK: [[ADDR:[0-9a-z]*]]  0 NOTYPE  GLOBAL DEFAULT   4 foo
# CHECK: [[ADDR]]            0 NOTYPE  GLOBAL DEFAULT   4 bar

        .section        .text.f,"ax",@progbits
        .globl  foo
foo:
        retq

        .section        .text.g,"ax",@progbits
        .globl  bar
bar:
        retq
