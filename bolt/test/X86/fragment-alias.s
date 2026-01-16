## This test reproduces the issue where a fragment has the same address as
## parent function.
# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t
# RUN: llvm-bolt %t -o %t.out 2>&1 | FileCheck %s
# CHECK: BOLT-WARNING: fragment maps to the same function as parent: main/1(*2)
.type main, @function
.type main.cold, @function
main.cold:
main:
  ret
.size main, .-main
.size main.cold, .-main.cold
