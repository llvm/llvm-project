// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
struct x { int a[100]; };


void foo(struct x *P, struct x *Q) {
// CHECK-LABEL: @foo(
// CHECK:    call void @llvm.memcpy.p0.p0
  *P = *Q;
}

// CHECK: declare void @llvm.memcpy.p0.p0{{.*}}(ptr noalias writeonly captures(none), ptr noalias readonly

void bar(struct x *P, struct x *Q) {
// CHECK-LABEL: @bar(
// CHECK:    call void @llvm.memcpy.p0.p0
  __builtin_memcpy(P, Q, sizeof(struct x));
}
