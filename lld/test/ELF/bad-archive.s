// REQUIRES: x86

// Check bad archive error reporting with --whole-archive
// and without it.
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: echo "!<arch>" > %t.bad.a
// RUN: echo "bad archive" >> %t.bad.a
// RUN: not ld.lld %t.o %t.bad.a -o %t 2>&1 | FileCheck %s
// RUN: not ld.lld %t.o --whole-archive %t.bad.a -o %t 2>&1 | FileCheck %s
// CHECK: {{.*}}.bad.a: failed to parse archive

.globl _start
_start:
