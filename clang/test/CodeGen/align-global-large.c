// PR13606 - Clang crashes with large alignment attribute
// RUN: %clang -S -emit-llvm %s -o - | FileCheck %s

// CHECK: x
// CHECK: align
// CHECK: 1048576
volatile char x[4000] __attribute__((aligned(0x100000)));

int
main (int argc, char ** argv) {
  // CHECK: y
  // CHECK: align
  // CHECK: 1048576
  volatile char y[4000] __attribute__((aligned(0x100000)));

  return y[argc];
}

