// RUN: %clangxx -fPIC -c -o %t.o %s
// RUN: %llvm_jitlink %t.o | FileCheck %s
// RUN: %clangxx -fPIC -emit-llvm -c -o %t.bc %s
// RUN: %lli_orc_jitlink -relocation-model=pic %t.bc | FileCheck %s

// CHECK: constructor
// CHECK-NEXT: main
// CHECK-NEXT: destructor

#include <stdio.h>

struct Global {
  Global() { puts("constructor"); }
  ~Global() { puts("destructor"); }
} G;

int main() { puts("main"); }
