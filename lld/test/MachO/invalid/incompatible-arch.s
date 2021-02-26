# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: not %lld -arch x86_64 -lSystem %t.o -o /dev/null 2>&1 | FileCheck %s -DFILE=%t.o
# CHECK: error: {{.*}}[[FILE]] has architecture arm64 which is incompatible with target architecture x86_64

.globl _main
_main:
  ret
