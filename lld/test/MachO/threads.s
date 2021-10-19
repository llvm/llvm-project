# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

## A positive integer is allowed.
# RUN: %lld --threads=1 %t.o -o /dev/null
# RUN: %lld --threads=2 %t.o -o /dev/null

# RUN: not %lld --threads=all %t.o -o /dev/null 2>&1 | FileCheck %s -DN=all
# RUN: not %lld --threads=0 %t.o -o /dev/null 2>&1 | FileCheck %s -DN=0
# RUN: not %lld --threads=-1 %t.o -o /dev/null 2>&1 | FileCheck %s -DN=-1

# CHECK: error: --threads=: expected a positive integer, but got '[[N]]'

.globl _main
_main:
  ret
