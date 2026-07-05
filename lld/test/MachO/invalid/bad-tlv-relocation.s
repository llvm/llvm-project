# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o /dev/null %t.o 2>&1 | FileCheck %s -DFILE=%t.o

# CHECK: [[FILE]]:(symbol _main+0x3): TLV relocation requires that symbol _foo be thread-local

.text
.globl _main
_main:
  leaq _foo@TLVP(%rip), %rax
  ret

.data
_foo:
