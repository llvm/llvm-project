// Test without serialization:
// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

void variadic(int, ...);

void test_floating_promotion(__fp16 *f16, float f32, double f64) {
  variadic(3, *f16, f32, f64);

// CHECK: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
// CHECK-NEXT: '__fp16'

// CHECK: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
// CHECK-NEXT: 'float'
}
