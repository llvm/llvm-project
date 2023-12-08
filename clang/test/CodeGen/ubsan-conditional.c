// RUN: %clang_cc1 %s -emit-llvm -fsanitize=float-divide-by-zero -o - | FileCheck %s

_Bool b;
// CHECK: @f(
double f(void) {
  // CHECK: %[[B:.*]] = load {{.*}} @b
  // CHECK: %[[COND:.*]] = trunc {{.*}} %[[B]] to i1
  // CHECK: br i1 %[[COND]]
  return b ? 0.0 / 0.0 : 0.0;
}
