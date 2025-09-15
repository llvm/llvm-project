// REQUIRES: x86
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
// RUN: echo > file.bin

// RUN: not ld.lld a.o --format=binary file.bin 2>&1 | FileCheck %s
// RUN: not ld.lld a.o --format binary file.bin 2>&1 | FileCheck %s

// CHECK:      error: duplicate symbol: _binary_file_bin_start
// CHECK-NEXT: defined in {{.*}}.o
// CHECK-NEXT: defined in file.bin

.globl  _binary_file_bin_start
_binary_file_bin_start:
  .long 0
