// RUN: %clang_cc1 -triple i386-unknown-unknown -fexceptions -emit-llvm %s -o - | FileCheck %s
int c(void) __attribute__((const));
int p(void) __attribute__((pure));
int t(void);

// CHECK: define{{.*}} i32 @_Z1fv() [[TF:#[0-9]+]] {
int f(void) {
  // CHECK: call i32 @_Z1cv() [[NUW_RN_CALL:#[0-9]+]]
  // CHECK: call i32 @_Z1pv() [[NUW_RO_CALL:#[0-9]+]]
  return c() + p() + t();
}

// CHECK: declare i32 @_Z1cv() [[NUW_RN:#[0-9]+]]
// CHECK: declare i32 @_Z1pv() [[NUW_RO:#[0-9]+]]
// CHECK: declare i32 @_Z1tv() [[TF2:#[0-9]+]]

// CHECK: attributes [[TF]] = { {{.*}} }
// CHECK: attributes [[NUW_RN]] = { nounwind readnone{{.*}} }
// CHECK: attributes [[NUW_RO]] = { nounwind readonly{{.*}} }
// CHECK: attributes [[TF2]] = { {{.*}} }
// CHECK: attributes [[NUW_RN_CALL]] = { nounwind readnone }
// CHECK: attributes [[NUW_RO_CALL]] = { nounwind readonly }
