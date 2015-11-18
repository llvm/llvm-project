// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-ar rcs %t.a %t.o
// RUN: not ld.lld -o %t2 -u _start %t.a 2>&1 | FileCheck %s

// CHECK: -m or at least a .o file required

.global _start
_start:
