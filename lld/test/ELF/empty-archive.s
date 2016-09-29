// RUN: llvm-ar rc %t.a
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld -shared %t.o %t.a -o t 2>&1 | FileCheck %s

// CHECK: has no symbol.
