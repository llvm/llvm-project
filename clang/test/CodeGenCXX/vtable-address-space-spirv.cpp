// SPIR-V only allows an addrspacecast into the generic address space, so the
// components of a vtable, which hold the addresses of functions, cannot live in
// the default globals address space there.

// RUN: %clang_cc1 %s -triple=spirv64 -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=spirv32 -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=spir64-unknown-unknown -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=spir-unknown-unknown -std=c++11 -emit-llvm -o - | FileCheck %s

// Functions already live in the generic address space on this target, so a cast
// into the globals address space is legal and the layout is left alone.
// RUN: %clang_cc1 %s -triple=spirv64-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s --check-prefix=AMDGCNSPIRV

struct A {
  virtual void f();
  virtual void g();
  virtual void h();
};

void A::f() {}

// The vtable itself stays a global, only its components become generic.
// CHECK: @_ZTV1A ={{.*}}addrspace(1) constant { [5 x ptr addrspace(4)] } { [5 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTI1A to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @_ZN1A1fEv to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @_ZN1A1gEv to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @_ZN1A1hEv to ptr addrspace(4))]
// CHECK: @_ZTI1A ={{.*}}addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) }

// AMDGCNSPIRV: @_ZTV1A ={{.*}}addrspace(1) constant { [5 x ptr addrspace(1)] } { [5 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) @_ZTI1A, ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1A1fEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1A1gEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr addrspace(4) @_ZN1A1hEv to ptr addrspace(1))]

void call(A *a) { a->g(); }

// The vtable pointer stays a globals-address-space pointer, the slot it points
// at holds a generic pointer.
// CHECK-LABEL: define {{.*}}@_Z4callP1A
// CHECK: %[[VT:.*]] = load ptr addrspace(1),
// CHECK: %[[SLOT:.*]] = getelementptr inbounds ptr addrspace(4), ptr addrspace(1) %[[VT]], i64 1
// CHECK: %[[FN:.*]] = load ptr addrspace(4), ptr addrspace(1) %[[SLOT]]
// CHECK: call {{.*}}addrspace(4) void %[[FN]]

// AMDGCNSPIRV-LABEL: define {{.*}}@_Z4callP1A
// AMDGCNSPIRV: %[[VT:.*]] = load ptr addrspace(1),
// AMDGCNSPIRV: %[[SLOT:.*]] = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %[[VT]], i64 1
// AMDGCNSPIRV: %[[FN:.*]] = load ptr addrspace(1), ptr addrspace(1) %[[SLOT]]
