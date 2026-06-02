// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fstrict-flex-arrays=3 -fsanitize=array-bounds -fsanitize-ignorelist=%t/ignore.list %t/test.c -o - | FileCheck %s --check-prefix=CHECK-IGNORED
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fstrict-flex-arrays=3 -fsanitize=array-bounds -fsanitize-ignorelist=%t/ignore.list %t/test2.c -o - | FileCheck %s --check-prefix=CHECK-NOT-IGNORED

//--- ignore.list
ignorefamsrc:*test.h

//--- test.h
struct Three {
  int ignored;
  int a[3];
  int b[3];
};

//--- test.c
#include "test.h"
// CHECK-IGNORED-LABEL: define {{.*}} @{{.*}}test_three_a{{.*}}(
int test_three_a(struct Three *p, int i) {
  // CHECK-IGNORED: call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i];
}

// CHECK-IGNORED-LABEL: define {{.*}} @{{.*}}test_three_b{{.*}}(
int test_three_b(struct Three *p, int i) {
  // CHECK-IGNORED-NOT: call void @__ubsan_handle_out_of_bounds_abort(
  return p->b[i];
}

//--- test2.h
struct Three {
  int ignored;
  int a[3];
};

//--- test2.c
#include "test2.h"
// CHECK-NOT-IGNORED-LABEL: define {{.*}} @{{.*}}test_three2_a{{.*}}(
int test_three2_a(struct Three *p, int i) {
  // CHECK-NOT-IGNORED: call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i];
}

// CHECK-NOT-IGNORED-LABEL: define {{.*}} @{{.*}}test_three2_b{{.*}}(
int test_three2_b(struct Three *p, int i) {
  // CHECK-NOT-IGNORED: call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i];
}
