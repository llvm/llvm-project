// RUN: %clang_cc1 -w -emit-llvm %s -O1 -o - | FileCheck %s
// Check for bug compatibility with gcc.

const int x __attribute((weak)) = 123;

int* f(void) {
  return &x;
}

int g(void) {
  // CHECK: ret i32 123
  return *f();
}
