// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o

// Creates a R_AARCH64_ABS64 relocation against _foo. It will be used on a
// shared object to check for a dynamic relocation creation.
.globl _foo
_foo:
   ret
_foo_init_array:
  .xword _foo

// RUN: ld.lld -shared -o %t.so %t.o
// RUN: llvm-readobj -symbols -dyn-relocations %t.so | FileCheck %s

// CHECK:       Dynamic Relocations {
// CHECK-NEXT:      {{.*}} R_AARCH64_RELATIVE - [[FOO_ADDR:[0-9xa-f]+]]

// CHECK:       Symbols [
// CHECK:         Symbol {
// CHECK:           Name: _foo
// CHECK:           Value: [[FOO_ADDR]]
