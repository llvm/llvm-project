// RUN: %clang_cc1 -triple avr -emit-llvm %s -o - | FileCheck %s

int main(void) {
  int (*p)(void);
  return 0;
}

// CHECK: %p = alloca i16 () addrspace(1)*
