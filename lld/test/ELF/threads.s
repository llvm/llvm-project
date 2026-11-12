# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## A positive integer is allowed.
# RUN: ld.lld --threads=1 %t.o -o /dev/null
# RUN: ld.lld --threads=2 %t.o -o /dev/null

# RUN: not ld.lld --threads=all %t.o -o /dev/null 2>&1 | FileCheck %s -DN=all
# RUN: not ld.lld --threads=0 %t.o -o /dev/null 2>&1 | FileCheck %s -DN=0
# RUN: not ld.lld --threads=-1 %t.o -o /dev/null 2>&1 | FileCheck %s -DN=-1

# CHECK: error: --threads: expected a positive integer, but got '[[N]]'
