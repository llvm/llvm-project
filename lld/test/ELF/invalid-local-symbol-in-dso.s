# REQUIRES: x86

# We used to crash on this
# RUN: not ld.lld %p/Inputs/local-symbol-in-dso.so -o /dev/null 2>&1 | FileCheck %s
# CHECK: error: {{.*}}local-symbol-in-dso.so: invalid local symbol 'foo' in global part of symbol table

# RUN: llvm-mc %s -o %t.o -filetype=obj -triple x86_64-pc-linux
# RUN: not ld.lld %t.o %p/Inputs/local-symbol-in-dso.so -o /dev/null

.globl main
main:
  movq foo@GOTTPOFF(%rip), %rax
