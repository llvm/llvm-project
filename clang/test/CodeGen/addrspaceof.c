// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

int test_default(int *p) {
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @test_default(
// CHECK: ret i32 0

int test_address_space(int __attribute__((address_space(4))) *p) {
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @test_address_space(
// CHECK: ret i32 16777220

int __attribute__((address_space(7))) *side_effect(void);

int test_unevaluated(void) {
  return __addrspaceof(*side_effect());
}

// CHECK-LABEL: define{{.*}} i32 @test_unevaluated(
// CHECK-NOT: call
// CHECK: ret i32 16777223

int array_default[4];
int __attribute__((address_space(5))) array_address_space[4];

int test_array_default(void) {
  return __addrspaceof(array_default);
}

// CHECK-LABEL: define{{.*}} i32 @test_array_default(
// CHECK: ret i32 0

int test_array_address_space(void) {
  return __addrspaceof(array_address_space);
}

// CHECK-LABEL: define{{.*}} i32 @test_array_address_space(
// CHECK: ret i32 16777221
