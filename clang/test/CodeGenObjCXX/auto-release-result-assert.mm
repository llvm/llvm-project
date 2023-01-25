// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

// CHECK-LABEL: define{{.*}} ptr @_Z4foo1i(
// CHECK: %[[CALL:[a-z0-9]+]] = call noundef ptr @_Z4foo0i
// CHECK: ret ptr %[[CALL]]

// CHECK-LABEL: define{{.*}} ptr @_ZN2S22m1Ev(
// CHECK: %[[CALL:[a-z0-9]+]] = call noundef ptr @_Z4foo0i
// CHECK: ret ptr %[[CALL]]

// CHECK-LABEL: define internal noundef ptr @Block1_block_invoke(
// CHECK: %[[CALL:[a-z0-9]+]] = call noundef ptr @_Z4foo0i
// CHECK: ret ptr %[[CALL]]

struct S1;

typedef __attribute__((NSObject)) struct __attribute__((objc_bridge(id))) S1 * S1Ref;

S1Ref foo0(int);

struct S2 {
  S1Ref m1();
};

S1Ref foo1(int a) {
  return foo0(a);
}

S1Ref S2::m1() {
  return foo0(0);
}

S1Ref (^Block1)(void) = ^{
  return foo0(0);
};
