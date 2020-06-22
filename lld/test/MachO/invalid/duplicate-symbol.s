# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t-dup.o
# RUN: not lld -flavor darwinnew -o /dev/null %t-dup.o %t.o 2>&1 | FileCheck %s

# CHECK: error: duplicate symbol: _main

.text
.global _main
_main:
  mov $0, %rax
  ret
