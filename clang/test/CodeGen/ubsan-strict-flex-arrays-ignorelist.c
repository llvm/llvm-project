// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fstrict-flex-arrays=3 -fsanitize=array-bounds -fsanitize-ignorelist=%t/ignore.list %t/test.c -o - | FileCheck %s --check-prefix=CHECK-IGNORED
// RUN: %clang_cc1 -emit-llvm -triple x86_64 -fstrict-flex-arrays=3 -fsanitize=array-bounds -fsanitize-ignorelist=%t/ignore.list %t/test2.c -o - | FileCheck %s --check-prefix=CHECK-NOT-IGNORED

//--- ignore.list
ignorefamsrc:*test.c

//--- test.c
struct Three {
  int ignored;
  int a[3];
};

// CHECK-IGNORED-LABEL: define {{.*}} @{{.*}}test_three{{.*}}(
int test_three(struct Three *p, int i) {
  // CHECK-IGNORED-NOT: call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i];
}

//--- test2.c
struct Three {
  int ignored;
  int a[3];
};

// CHECK-NOT-IGNORED-LABEL: define {{.*}} @{{.*}}test_three_2{{.*}}(
int test_three_2(struct Three *p, int i) {
  // CHECK-NOT-IGNORED: call void @__ubsan_handle_out_of_bounds_abort(
  return p->a[i];
}
