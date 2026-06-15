# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: not ld.lld %t.o --irpgo-profile=%t.missing.profdata --bp-startup-sort=function 2>&1 | FileCheck %s

# CHECK: error: No such file or directory

.globl _start
_start:
  ret
