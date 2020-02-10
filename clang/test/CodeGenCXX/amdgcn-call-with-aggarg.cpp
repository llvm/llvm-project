// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -O3 -disable-llvm-passes -o - %s | FileCheck %s

struct A {
  float x, y, z, w;
};

void foo(A a);

// CHECK-LABEL: @_Z4testv
// CHECK: %[[lvar:.*]] = alloca %struct.A, align 4, addrspace(5)
// CHECK: %[[atmp:.*]] = alloca %struct.A, align 4, addrspace(5)
// CHECK: %[[lcst:.*]] = bitcast %struct.A addrspace(5)* %[[lvar]] to i8 addrspace(5)*
// CHECK: call void @llvm.lifetime.start.p5i8(i64 16, i8 addrspace(5)* %[[lcst]]
// CHECK: %[[acst:.*]] = bitcast %struct.A addrspace(5)* %[[atmp]] to i8 addrspace(5)*
// CHECK: call void @llvm.lifetime.start.p5i8(i64 16, i8 addrspace(5)* %[[acst]]
void test() {
  A a;
  foo(a);
}
