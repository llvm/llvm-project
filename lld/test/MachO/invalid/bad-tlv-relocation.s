# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o /dev/null %t.o 2>&1 | FileCheck %s -DFILE=%t.o

# CHECK: TLV relocation requires that variable be thread-local for `_foo' in [[FILE]]:(__text)

.text
.globl _main
_main:
  leaq _foo@TLVP(%rip), %rax
  ret

.data
_foo:
