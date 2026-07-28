// RUN: %clang_cc1 %s -triple=spirv64 -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=spirv32 -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=spir64-unknown-unknown -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=spir-unknown-unknown -std=c++11 -emit-llvm -o - | FileCheck %s

// Functions are already generic here, so the components stay global.
// RUN: %clang_cc1 %s -triple=spirv64-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s --check-prefix=AMDGCNSPIRV

namespace std {
struct type_info;
} // namespace std

struct A {
  virtual void f();
  virtual void g();
};

void A::f() {}

// CHECK: @_ZTV1A ={{.*}}addrspace(1) constant { [4 x ptr addrspace(4)] } { [4 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTI1A to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @_ZN1A1fEv to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @_ZN1A1gEv to ptr addrspace(4))]
// CHECK: @_ZTI1A ={{.*}}addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) }

// AMDGCNSPIRV: @_ZTV1A ={{.*}}addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) @_ZTI1A, ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1A1fEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1A1gEv to ptr addrspace(1))]

void call(A *a) { a->g(); }

// CHECK-LABEL: define {{.*}}@_Z4callP1A
// CHECK: %[[VT:.*]] = load ptr addrspace(1),
// CHECK: %[[SLOT:.*]] = getelementptr inbounds ptr addrspace(4), ptr addrspace(1) %[[VT]], i64 1
// CHECK: %[[FN:.*]] = load ptr addrspace(4), ptr addrspace(1) %[[SLOT]]
// CHECK: call {{.*}}addrspace(4) void %[[FN]]

// AMDGCNSPIRV-LABEL: define {{.*}}@_Z4callP1A
// AMDGCNSPIRV: %[[VT:.*]] = load ptr addrspace(1),
// AMDGCNSPIRV: %[[SLOT:.*]] = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %[[VT]], i64 1
// AMDGCNSPIRV: %[[FN:.*]] = load ptr addrspace(1), ptr addrspace(1) %[[SLOT]]

const std::type_info &ti(A *a) { return typeid(*a); }

// CHECK-LABEL: define {{.*}}@_Z2tiP1A
// CHECK: %[[TVT:.*]] = load ptr addrspace(1),
// CHECK: %[[TSLOT:.*]] = getelementptr inbounds ptr addrspace(4), ptr addrspace(1) %[[TVT]], i64 -1
// CHECK: %[[TI:.*]] = load ptr addrspace(4), ptr addrspace(1) %[[TSLOT]]
// CHECK: addrspacecast ptr addrspace(4) %[[TI]] to ptr

// AMDGCNSPIRV-LABEL: define {{.*}}@_Z2tiP1A
// AMDGCNSPIRV: %[[TVT:.*]] = load ptr addrspace(1),
// AMDGCNSPIRV: %[[TSLOT:.*]] = getelementptr inbounds ptr addrspace(4), ptr addrspace(1) %[[TVT]], i64 -1
// AMDGCNSPIRV: %[[TI:.*]] = load ptr addrspace(4), ptr addrspace(1) %[[TSLOT]]
