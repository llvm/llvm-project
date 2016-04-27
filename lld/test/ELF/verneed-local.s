# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: not ld.lld %t.o %S/Inputs/verneed1.so -o %t 2>&1 | FileCheck %s

# CHECK: undefined symbol: f3 in
.globl _start
_start:
call f3
