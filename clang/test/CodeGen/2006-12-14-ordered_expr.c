// RUN: %clang_cc1 -O3 -emit-llvm %s -o - | FileCheck %s

int test2(float X, float Y) {
  // CHECK: fcmp ord float %X, %Y
  return !__builtin_isunordered(X, Y);
}
