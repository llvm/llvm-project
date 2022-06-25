// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fsanitize=array-bounds %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-STRICT-0
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fsanitize=array-bounds -fstrict-flex-arrays=1 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-STRICT-1
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fsanitize=array-bounds -fstrict-flex-arrays=2 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-STRICT-2

// Before flexible array member was added to C99, many projects use a
// one-element array as the last emember of a structure as an alternative.
// E.g. https://github.com/python/cpython/issues/94250
// Suppress such errors with -fstrict-flex-arrays=0.
struct One {
  int a[1];
};
struct Two {
  int a[2];
};
struct Three {
  int a[3];
};

// CHECK-LABEL: define {{.*}} @test_one(
int test_one(struct One *p, int i) {
  // CHECK-STRICT-0-NOT: @__ubsan
  // CHECK-STRICT-1-NOT: @__ubsan
  // CHECK-STRICT-2:     call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i] + (p->a)[i];
}

// CHECK-LABEL: define {{.*}} @test_two(
int test_two(struct Two *p, int i) {
  // CHECK-STRICT-0:     call void @__ubsan_handle_out_of_bounds_abort(
  // CHECK-STRICT-1:     call void @__ubsan_handle_out_of_bounds_abort(
  // CHECK-STRICT-2:     call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i] + (p->a)[i];
}

// CHECK-LABEL: define {{.*}} @test_three(
int test_three(struct Three *p, int i) {
  // CHECK-STRICT-0:     call void @__ubsan_handle_out_of_bounds_abort(
  // CHECK-STRICT-1:     call void @__ubsan_handle_out_of_bounds_abort(
  // CHECK-STRICT-2:     call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i] + (p->a)[i];
}
