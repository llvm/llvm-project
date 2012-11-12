// RUN: %clang_cc1 -triple i386-unknown-unknown -fexceptions -emit-llvm %s -o - | FileCheck %s
int c(void) __attribute__((const));
int p(void) __attribute__((pure));
int t(void);

// CHECK: define i32 @_Z1fv() {
int f(void) {
  // CHECK: call i32 @_Z1cv() nounwind readnone
  // CHECK: call i32 @_Z1pv() nounwind readonly
  return c() + p() + t();
}

// CHECK: declare i32 @_Z1cv() nounwind readnone
// CHECK: declare i32 @_Z1pv() nounwind readonly
// CHECK-NOT: declare i32 @_Z1tv() nounwind
