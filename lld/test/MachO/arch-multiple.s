# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t.o %s
# RUN: not %lld -o %t.out -arch_multiple %t.o 2>&1 | FileCheck %s

# CHECK: error: undefined symbol for arch x86_64: _foo

.globl _main
_main:
  callq _foo
  ret
