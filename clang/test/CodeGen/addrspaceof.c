// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT

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

int test_vla_bound_is_unevaluated(void) {
  int bound = 1;
  (void)__addrspaceof(int[bound++]);
  return bound;
}

// CHECK-LABEL: define{{.*}} i32 @test_vla_bound_is_unevaluated(
// CHECK-NOT: add
// CHECK: ret i32
// OPT-LABEL: define{{.*}} i32 @test_vla_bound_is_unevaluated(
// OPT: ret i32 1

_Static_assert(__addrspaceof((int __attribute__((address_space(12)))){100}) ==
                   __ADDRSPACE_TARGET(12),
               "compound literal address space");

enum { named_constant };
_Static_assert(__addrspaceof(named_constant) == __ADDRSPACE_DEFAULT,
               "enumerator address space");

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
