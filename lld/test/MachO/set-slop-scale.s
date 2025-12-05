# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-darwin %s -o %t.o
# RUN: %lld -o /dev/null %t.o --slop_scale=1
# RUN: not %lld -o /dev/null %t.o --slop_scale=-1 2>&1 | FileCheck %s
# CHECK: error: --slop_scale=: expected a non-negative integer, but got '-1'

.text
.global _main
_main:
  mov $0, %rax
  ret
