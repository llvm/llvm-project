// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fstrict-flex-arrays=3 -fsanitize=array-bounds -fsanitize-ignorelist=%t/ignore.list %t/test.c -o - | FileCheck %s --check-prefix=CHECK

//--- ignore.list
ignorelastmembersrc:*test.h

//--- test.h
struct StructInTestH {
  int ignored;
  int a[3];
  int b[3];
};

//--- test2.h
struct StructInTest2H {
  int ignored;
  int a[3];
  int b[3];
};

//--- test.c
#include "test.h"
#include "test2.h"

// CHECK-LABEL: define {{.*}} @{{.*}}test_struct_in_test_h_a{{.*}}(
int test_struct_in_test_h_a(struct StructInTestH *p, int i) {
  // CHECK: call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i];
}

// CHECK-LABEL: define {{.*}} @{{.*}}test_struct_in_test_h_b{{.*}}(
int test_struct_in_test_h_b(struct StructInTestH *p, int i) {
  // CHECK-NOT: call void @__ubsan_handle_out_of_bounds_abort(
  return p->b[i];
}


// CHECK-LABEL: define {{.*}} @{{.*}}test_struct_in_test2_h_a{{.*}}(
int test_struct_in_test2_h_a(struct StructInTest2H *p, int i) {
  // CHECK: call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i];
}

// CHECK-LABEL: define {{.*}} @{{.*}}test_struct_in_test2_h_b{{.*}}(
int test_struct_in_test2_h_b(struct StructInTest2H *p, int i) {
  // CHECK: call void @__ubsan_handle_out_of_bounds_abort(
  return p->b[i];
}
