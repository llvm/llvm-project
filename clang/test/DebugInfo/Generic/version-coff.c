// REQUIRES: x86-registered-target
// RUN: %clang --target=x86_64-windows -g -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang --target=x86_64-windows -S -emit-llvm -o - %s | FileCheck %s
int main (void) {
  return 0;
}

// CHECK:  i32 2, !"Debug Info Version", i32 3}
