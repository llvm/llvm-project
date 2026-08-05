// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s
// RUN: %clang_cc1 %s -triple=spirv64-amd-amdhsa -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s --check-prefix=WITH-NONZERO-DEFAULT-AS

struct X { };

const X g();

void f() {
  try {
    throw g();
    // CHECK: ptr addrspace(1) @_ZTI1X
  } catch (const X x) {
    // CHECK: catch ptr addrspace(1) @_ZTI1X
    // CHECK: call i32 @llvm.eh.typeid.for.p0(ptr addrspacecast (ptr addrspace(1) @_ZTI1X to ptr))
    // WITH-NONZERO-DEFAULT-AS: call{{.*}} i32 @llvm.eh.typeid.for.p4(ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTI1X to ptr addrspace(4)))
  }
}

void h() {
  try {
    throw "ABC";
    // CHECK: ptr addrspace(1) @_ZTIPKc
  } catch (char const(&)[4]) {
    // CHECK: catch ptr addrspace(1) @_ZTIA4_c
    // CHECK: call i32 @llvm.eh.typeid.for.p0(ptr addrspacecast (ptr addrspace(1) @_ZTIA4_c to ptr))
    // WITH-NONZERO-DEFAULT-AS: call{{.*}} i32 @llvm.eh.typeid.for.p4(ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTIA4_c to ptr addrspace(4)))
  }
}
