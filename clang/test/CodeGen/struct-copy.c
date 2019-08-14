// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
struct x { int a[100]; };


void foo(struct x *P, struct x *Q) {
// CHECK-LABEL: @foo(
// CHECK:    call void @llvm.memcpy.p0i8.p0i8
  *P = *Q;
}

// CHECK: declare void @llvm.memcpy.p0i8.p0i8{{.*}}(i8* noalias nocapture writeonly, i8* noalias nocapture readonly

void bar(struct x *P, struct x *Q) {
// CHECK-LABEL: @bar(
// CHECK:    call void @llvm.memcpy.p0i8.p0i8
  __builtin_memcpy(P, Q, sizeof(struct x));
}
