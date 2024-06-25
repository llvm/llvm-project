// RUN: %clang --target=aarch64 %s -S -emit-llvm -o - | FileCheck %s

_Complex long double a, b, c, d;
void test_fp128_compound_assign(void) {
  // CHECK: call { fp128, fp128 } @__multc3
  a *= b;
  // CHECK: call { fp128, fp128 } @__divtc3
  c /= d;
}
