# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -filelist nonexistent %t.o -o %t 2>&1 | FileCheck %s
# CHECK: cannot open nonexistent: {{N|n}}o such file or directory

.globl _main

_main:
  ret
