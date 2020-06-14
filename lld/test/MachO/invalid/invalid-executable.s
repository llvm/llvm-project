# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: not lld -flavor darwinnew -o /dev/null %t 2>&1 | FileCheck %s -DFILE=%t
# CHECK: error: [[FILE]]: unhandled file type

.text
.global _main
_main:
  mov $0, %rax
  ret
