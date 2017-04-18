// RUN: %clang_cc1 -fsanitize=null -emit-llvm %s -o - | FileCheck %s

struct A {
  int a[2];
  int b;
};

// CHECK-LABEL: @f1
int *f1() {
// CHECK-NOT: __ubsan_handle_type_mismatch
// CHECK: ret
// CHECK-SAME: getelementptr inbounds (%struct.A, %struct.A* null, i32 0, i32 1)
  return &((struct A *)0)->b;
}
