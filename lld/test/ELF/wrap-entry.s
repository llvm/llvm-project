// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

// RUN: ld.lld -o %t.exe %t.o -wrap=_start
// RUN: llvm-readobj --symbols -h %t.exe | FileCheck %s

/// Note, ld.bfd uses _start as the _entry.

// CHECK:      Entry: [[ADDR:[0-9A-F]+]]
// CHECK:      Name: __wrap__start
// CHECK-NEXT: Value: [[ADDR]]

.global _start, __wrap__start
_start:
  nop
__wrap__start:
  nop
