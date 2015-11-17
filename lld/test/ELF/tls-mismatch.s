// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/tls-mismatch.s -o %t2
// RUN: not ld.lld2 %t %t2 -o %t3 2>&1 | FileCheck %s
// CHECK: TLS attribute mismatch for symbol: tlsvar

.globl _start,tlsvar
_start:
  movl tlsvar,%edx
