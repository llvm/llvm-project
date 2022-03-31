// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: cd %t.dir
// RUN: echo > file.bin

// RUN: not ld.lld %t.o --format=binary file.bin -o /dev/null 2>&1 | FileCheck %s
// RUN: not ld.lld %t.o --format binary file.bin -o /dev/null 2>&1 | FileCheck %s

// CHECK:      duplicate symbol: _binary_file_bin_start
// CHECK-NEXT: defined in {{.*}}.o
// CHECK-NEXT: defined in <internal>

.globl  _binary_file_bin_start
_binary_file_bin_start:
  .long 0
