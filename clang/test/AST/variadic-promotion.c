// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

void variadic(int, ...);

void test_floating_promotion(__fp16 *f16, float f32, double f64) {
  variadic(3, *f16, f32, f64);

// CHECK: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
// CHECK-NEXT: '__fp16'

// CHECK: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
// CHECK-NEXT: 'float'
}
