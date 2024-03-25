// RUN: %clang_cc1 -std=c++20 -triple=amdgcn-amd-amdhsa %s -emit-llvm -o - | FileCheck %s --check-prefix=ITANIUM --implicit-check-not=DoNotEmit

// FIXME: The MSVC ABI rule in use here was discussed with MS folks prior to
// them implementing virtual consteval functions, but we do not know for sure
// if this is the ABI rule they will use.

// ITANIUM-DAG: @_ZTV1A = {{.*}} addrspace(1) constant { [2 x ptr addrspace(1)] } {{.*}} null, {{.*}} @_ZTI1A
struct A {
  virtual consteval void DoNotEmit_f() {}
};
// ITANIUM-DAG: @a = addrspace(1) global { {{.*}} ptr addrspace(1) @_ZTV1A,
A a;

// ITANIUM-DAG: @_ZTV1B = {{.*}} addrspace(1) constant { [4 x ptr addrspace(1)] } {{.*}} addrspace(1) null, ptr addrspace(1) @_ZTI1B, ptr addrspace(1) addrspacecast (ptr @_ZN1B1fEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN1B1hEv to ptr addrspace(1))
struct B {
  virtual void f() {}
  virtual consteval void DoNotEmit_g() {}
  virtual void h() {}
};
// ITANIUM-DAG: @b = addrspace(1) global { {{.*}} @_ZTV1B,
B b;

// ITANIUM-DAG: @_ZTV1C = {{.*}} addrspace(1) constant { [4 x ptr addrspace(1)] } {{.*}} addrspace(1) null, ptr addrspace(1) @_ZTI1C, ptr addrspace(1) addrspacecast (ptr @_ZN1CD1Ev to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN1CD0Ev to ptr addrspace(1))
struct C {
  virtual ~C() = default;
  virtual consteval C &operator=(const C&) = default;
};
// ITANIUM-DAG: @c = addrspace(1) global { {{.*}} @_ZTV1C,
C c;

// ITANIUM-DAG: @_ZTV1D = {{.*}} addrspace(1) constant { [4 x ptr addrspace(1)] } {{.*}} addrspace(1) null, ptr addrspace(1) @_ZTI1D, ptr addrspace(1) addrspacecast (ptr @_ZN1DD1Ev to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @_ZN1DD0Ev to ptr addrspace(1))
struct D : C {};
// ITANIUM-DAG: @d = addrspace(1) global { ptr addrspace(1) } { {{.*}} @_ZTV1D,
D d;

// ITANIUM-DAG: @_ZTV1E = {{.*}} addrspace(1) constant { [3 x ptr addrspace(1)] } {{.*}} addrspace(1) null, ptr addrspace(1) @_ZTI1E, ptr addrspace(1) addrspacecast (ptr @_ZN1E1fEv to ptr addrspace(1))
struct E { virtual void f() {} };
// ITANIUM-DAG: @e = addrspace(1) global { {{.*}} @_ZTV1E,
E e;

// ITANIUM-DAG: @_ZTV1F = {{.*}} addrspace(1) constant { [3 x ptr addrspace(1)] } {{.*}} addrspace(1) null, ptr addrspace(1) @_ZTI1F, ptr addrspace(1) addrspacecast (ptr @_ZN1E1fEv to ptr addrspace(1))
struct F : E { virtual consteval void DoNotEmit_g(); };
// ITANIUM-DAG: @f = addrspace(1) global { ptr addrspace(1) } { {{.*}} @_ZTV1F,
F f;
