# REQUIRES: aarch64, x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.arm64.o
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %s -o %t.arm64e.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %s -o %t.arm64_32.o

# RUN: %lld -arch x86_64 -lSystem -no_pie -o %t %t.x86_64.o
# RUN: not %lld -arch arm64 -lSystem -no_pie -o %t %t.arm64.o 2>&1 | FileCheck %s
# RUN: not %lld -arch arm64e -lSystem -no_pie -o %t %t.arm64e.o 2>&1 | FileCheck %s
# RUN: not %lld-watchos -arch arm64_32 -lSystem -no_pie -o %t %t.arm64_32.o 2>&1 | FileCheck %s

# CHECK: error: -no_pie ignored for arm64

.globl _main
_main:
  ret
