// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

/* WG14 N1391: Yes
 * Floating-point to int/_Bool conversions
 */

int neg_zero(void) {
  // CHECK: define{{.*}} i32 @neg_zero()
  return (_Bool)-0.0 ? -1 : 1; // Negative zero -> false
  // CHECK: ret i32 1
}

int pos_inf(void) {
  // CHECK: define{{.*}} i32 @pos_inf()
  return (_Bool)(1.0f / 0.0f) ? 1 : -1; // Positive inf -> true
  // CHECK: ret i32 1
}

int neg_inf(void) {
  // CHECK: define{{.*}} i32 @neg_inf()
  return (_Bool)(-1.0f / 0.0f) ? 1 : -1; // Negative inf -> true
  // CHECK: ret i32 1
}

int nan(void) {
  // CHECK: define{{.*}} i32 @nan()
  return (_Bool)(0.0f / 0.0f) ? 1 : -1; // NaN -> true
  // CHECK: ret i32 1
}
