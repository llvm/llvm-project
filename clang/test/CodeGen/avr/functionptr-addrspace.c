// RUN: %clang_cc1 -triple avr -emit-llvm %s -o - | FileCheck %s

int main() {
  int (*p)();
  return 0;
}

// CHECK: %p = alloca i16 (...) addrspace(1)*
