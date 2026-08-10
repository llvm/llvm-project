# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -filelist nonexistent %t.o -o %t 2>&1 | FileCheck -DMSG=%errc_ENOENT %s
# CHECK: cannot open nonexistent: [[MSG]]

.globl _main

_main:
  ret
