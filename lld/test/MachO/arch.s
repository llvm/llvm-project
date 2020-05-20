# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -arch x86_64 -o /dev/null %t.o
# RUN: not lld -flavor darwinnew -arch i386 -o /dev/null %t.o 2>&1 | FileCheck %s
# CHECK: error: missing or unsupported -arch i386

.text
.global _main
_main:
  mov $0, %rax
  ret
