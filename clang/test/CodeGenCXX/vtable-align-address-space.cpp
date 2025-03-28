// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s

struct A {
  virtual void f();
  virtual void g();
  virtual void h();
};

void A::f() {}

// CHECK: @_ZTV1A ={{.*}} unnamed_addr addrspace(1) constant { [5 x ptr addrspace(1)] } { [5 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) @_ZTI1A, ptr addrspace(1) addrspacecast (ptr @_ZN1A1fEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN1A1gEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN1A1hEv to ptr addrspace(1))]
// CHECK: @_ZTI1A ={{.*}} addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) getelementptr inbounds (ptr addrspace(1), ptr addrspace(1) @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr addrspace(1) @_ZTS1A }, align 8
// CHECK: @_ZTS1A ={{.*}} constant [3 x i8] c"1A\00", align 1
