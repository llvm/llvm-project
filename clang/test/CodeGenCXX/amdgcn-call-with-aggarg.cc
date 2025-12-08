// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -O3 -disable-llvm-passes -o - %s | FileCheck %s

struct A {
  float x, y, z, w;
};

void foo(A a);

// CHECK-LABEL: @_Z4testv
// CHECK:         [[A:%.*]] = alloca [[STRUCT_A:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[AGG_TMP:%.*]] = alloca [[STRUCT_A]], align 4, addrspace(5)
// CHECK-NEXT:    [[A_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[A]] to ptr
// CHECK-NEXT:    [[AGG_TMP_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[AGG_TMP]] to ptr
// CHECK-NEXT:    call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) [[A]]) #[[ATTR4:[0-9]+]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) [[AGG_TMP]]) #[[ATTR4]]
void test() {
  A a;
  foo(a);
}
