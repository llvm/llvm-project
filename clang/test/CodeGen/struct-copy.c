// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
struct x { int a[100]; };


void foo(struct x *P, struct x *Q) {
// CHECK-LABEL: @foo(
// CHECK:    call void @llvm.memcpy.p0i8.p0i8.i64
  *P = *Q;
}

// CHECK: declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

void bar(struct x *P, struct x *Q) {
// CHECK-LABEL: @bar(
// CHECK:    call void @llvm.memcpy.p0i8.p0i8.i64
  __builtin_memcpy(P, Q, sizeof(struct x));
}
