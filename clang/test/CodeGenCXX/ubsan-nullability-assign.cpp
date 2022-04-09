// RUN: %clang_cc1 -no-opaque-pointers -x c++ -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=nullability-assign | FileCheck %s

struct S1 {
  int *_Nonnull p;
};

struct S2 {
  S1 s1;
};

union U1 {
  S1 s1;
  S2 s2;
};

// CHECK-LABEL: define{{.*}} void @{{.*}}f1
void f1(int *p) {
  U1 u;

  // CHECK: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}} !nosanitize
  // CHECK: store
  u.s1.p = p;

  // CHECK: [[ICMP:%.*]] = icmp ne i32* {{.*}}, null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], {{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_type_mismatch{{.*}} !nosanitize
  // CHECK: store
  u.s2.s1.p = p;

  // CHECK-NOT: __ubsan_handle_type_mismatch
  // CHECK-NOT: store
  // CHECK: ret void
}
