// RUN: %clang_cc1 %s -O0 -fsanitize=shift-exponent -emit-llvm -std=c2x -triple=x86_64-unknown-linux -o - | FileCheck %s

// Checking that the code generation is using the unextended/untruncated
// exponent values and capping the values accordingly

// CHECK-LABEL: define{{.*}} i32 @test_left_variable
int test_left_variable(unsigned _BitInt(5) b, unsigned _BitInt(2) e) {
  // CHECK: [[E_REG:%.+]] = load [[E_SIZE:i2]]
  // CHECK: icmp ule [[E_SIZE]] [[E_REG]], -1
  return b << e;
}

// CHECK-LABEL: define{{.*}} i32 @test_right_variable
int test_right_variable(unsigned _BitInt(2) b, unsigned _BitInt(3) e) {
  // CHECK: [[E_REG:%.+]] = load [[E_SIZE:i3]]
  // CHECK: icmp ule [[E_SIZE]] [[E_REG]], 1
  return b >> e;
}

// Old code generation would give false positives on left shifts when:
//   value(e) > (width(b) - 1 % 2 ** width(e))
// CHECK-LABEL: define{{.*}} i32 @test_left_literal
int test_left_literal(unsigned _BitInt(5) b) {
  // CHECK-NOT: br i1 false, label %cont, label %handler.shift_out_of_bounds
  // CHECK: br i1 true, label %cont, label %handler.shift_out_of_bounds
  return b << 3uwb;
}

// Old code generation would give false positives on right shifts when:
//   (value(e) % 2 ** width(b)) < width(b)
// CHECK-LABEL: define{{.*}} i32 @test_right_literal
int test_right_literal(unsigned _BitInt(2) b) {
  // CHECK-NOT: br i1 true, label %cont, label %handler.shift_out_of_bounds
  // CHECK: br i1 false, label %cont, label %handler.shift_out_of_bounds
  return b >> 4uwb;
}
