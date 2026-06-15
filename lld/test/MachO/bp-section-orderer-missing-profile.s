# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -arch x86_64 -lSystem -e _main -o /dev/null %t.o --irpgo-profile=%t.missing.profdata --bp-startup-sort=function 2>&1 | FileCheck %s

# CHECK: error: No such file or directory

.text
.globl _main
_main:
  ret
