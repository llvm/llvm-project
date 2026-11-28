// RUN: %clangxx -fPIC -fexceptions -c -o %t.o %s
// RUN: %llvm_jitlink %t.o | FileCheck %s
// RUN: %clangxx -fPIC -fexceptions -emit-llvm -c -o %t.bc %s
// RUN: %lli_orc_jitlink -relocation-model=pic %t.bc | FileCheck %s

// CHECK: catch

#include <stdio.h>

int main() {
  try {
    throw 0;
  } catch (int X) {
    puts("catch");
    return X;
  }
  return 1;
}
