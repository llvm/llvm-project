// RUN: %clang_cc1 -triple x86_64 -emit-llvm -o - -fsanitize=numerical %s | FileCheck %s

// CHECK: Function Attrs: noinline nounwind optnone sanitize_numerical_stability
float add(float x, float y) {
  float z = x + y;
  return z;
}
