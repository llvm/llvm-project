// Check that constructors and destructors are run in the expected order.
//
// RUN: %clang -c -o %t.o %s
// RUN: %llvm_jitlink %t.o | FileCheck %s
//
// REQUIRES: system-linux && host-arch-compatible

// CHECK: constructor #1
// CHECK: constructor #2
// CHECK: constructor #N
// CHECK: destructor #N
// CHECK: destructor #2
// CHECK: destructor #1

#include <stdio.h>

int x = 0;

__attribute__((constructor(101))) void constructor1() {
  puts("constructor #1");
}
__attribute__((constructor(102))) void constructor2() {
  puts("constructor #2");
}
__attribute__((constructor)) void constructorN() { puts("constructor #N"); }

__attribute__((destructor(101))) void destructor1() { puts("destructor #1"); }
__attribute__((destructor(102))) void destructor2() { puts("destructor #2"); }
__attribute__((destructor)) void destructorN() { puts("destructor #N"); }

int main(void) { return 0; }