# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not lld -flavor darwinnew -Z -o %t %t.o 2>&1 | FileCheck %s -DBASENAME=%basename_t
# CHECK: error: undefined symbol _foo, referenced from [[BASENAME]]

.globl _main
.text
_main:
  callq _foo
  movq $0, %rax
  retq
