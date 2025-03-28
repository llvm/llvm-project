// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o - -triple x86_64-linux | FileCheck %s --check-prefix=ITANIUM --implicit-check-not=DoNotEmit
// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o - -triple x86_64-windows | FileCheck %s --check-prefix=MSABI --implicit-check-not=DoNotEmit

// FIXME: The MSVC ABI rule in use here was discussed with MS folks prior to
// them implementing virtual consteval functions, but we do not know for sure
// if this is the ABI rule they will use.

// ITANIUM-DAG: @_ZTV1A = {{.*}} constant { [2 x ptr] } {{.*}} null, {{.*}} @_ZTI1A
// MSABI-DAG: @[[A_VFTABLE:.*]] = {{.*}} constant { [1 x ptr] } {{.*}} @"??_R4A@@6B@"
struct A {
  virtual consteval void DoNotEmit_f() {}
};
// ITANIUM-DAG: @a = {{.*}}global %struct.A { {{.*}} @_ZTV1A,
// MSABI-DAG: @"?a@@3UA@@A" = {{.*}}global %struct.A { ptr @"??_7A@@6B@" }
A a;

// ITANIUM-DAG: @_ZTV1B = {{.*}} constant { [4 x ptr] } {{.*}} null, ptr @_ZTI1B, ptr @_ZN1B1fEv, ptr @_ZN1B1hEv
// MSABI-DAG: @[[B_VFTABLE:.*]] = {{.*}} constant { [3 x ptr] } {{.*}} @"??_R4B@@6B@", ptr @"?f@B@@UEAAXXZ", ptr @"?h@B@@UEAAXXZ"
struct B {
  virtual void f() {}
  virtual consteval void DoNotEmit_g() {}
  virtual void h() {}
};
// ITANIUM-DAG: @b = {{.*}}global %struct.B { {{.*}} @_ZTV1B,
// MSABI-DAG: @"?b@@3UB@@A" = {{.*}}global %struct.B { ptr @"??_7B@@6B@" }
B b;

// ITANIUM-DAG: @_ZTV1C = {{.*}} constant { [4 x ptr] } {{.*}} null, ptr @_ZTI1C, ptr @_ZN1CD1Ev, ptr @_ZN1CD0Ev
// MSABI-DAG: @[[C_VFTABLE:.*]] = {{.*}} constant { [2 x ptr] } {{.*}} @"??_R4C@@6B@", ptr @"??_GC@@UEAAPEAXI@Z"
struct C {
  virtual ~C() = default;
  virtual consteval C &operator=(const C&) = default;
};
// ITANIUM-DAG: @c = {{.*}}global %struct.C { {{.*}} @_ZTV1C,
// MSABI-DAG: @"?c@@3UC@@A" = {{.*}}global %struct.C { ptr @"??_7C@@6B@" }
C c;

// ITANIUM-DAG: @_ZTV1D = {{.*}} constant { [4 x ptr] } {{.*}} null, ptr @_ZTI1D, ptr @_ZN1DD1Ev, ptr @_ZN1DD0Ev
// MSABI-DAG: @[[D_VFTABLE:.*]] = {{.*}} constant { [2 x ptr] } {{.*}} @"??_R4D@@6B@", ptr @"??_GD@@UEAAPEAXI@Z"
struct D : C {};
// ITANIUM-DAG: @d = {{.*}}global { ptr } { {{.*}} @_ZTV1D,
// MSABI-DAG: @"?d@@3UD@@A" = {{.*}}global { ptr } { ptr @"??_7D@@6B@" }
D d;

// ITANIUM-DAG: @_ZTV1E = {{.*}} constant { [3 x ptr] } {{.*}} null, ptr @_ZTI1E, ptr @_ZN1E1fEv
// MSABI-DAG: @[[E_VFTABLE:.*]] = {{.*}} constant { [2 x ptr] } {{.*}} @"??_R4E@@6B@", ptr @"?f@E@@UEAAXXZ"
struct E { virtual void f() {} };
// ITANIUM-DAG: @e = {{.*}}global %struct.E { {{.*}} @_ZTV1E,
// MSABI-DAG: @"?e@@3UE@@A" = {{.*}}global %struct.E { ptr @"??_7E@@6B@" }
E e;

// ITANIUM-DAG: @_ZTV1F = {{.*}} constant { [3 x ptr] } {{.*}} null, ptr @_ZTI1F, ptr @_ZN1E1fEv
// MSABI-DAG: @[[F_VFTABLE:.*]] = {{.*}} constant { [2 x ptr] } {{.*}} @"??_R4F@@6B@", ptr @"?f@E@@UEAAXXZ"
struct F : E { virtual consteval void DoNotEmit_g(); };
// ITANIUM-DAG: @f = {{.*}}global { ptr } { {{.*}} @_ZTV1F,
// MSABI-DAG: @"?f@@3UF@@A" = {{.*}}global { ptr } { ptr @"??_7F@@6B@" }
F f;

// MSABI-DAG: @"??_7A@@6B@" = {{.*}} alias {{.*}} @[[A_VFTABLE]],
// MSABI-DAG: @"??_7B@@6B@" = {{.*}} alias {{.*}} @[[B_VFTABLE]],
// MSABI-DAG: @"??_7C@@6B@" = {{.*}} alias {{.*}} @[[C_VFTABLE]],
// MSABI-DAG: @"??_7D@@6B@" = {{.*}} alias {{.*}} @[[D_VFTABLE]],
// MSABI-DAG: @"??_7E@@6B@" = {{.*}} alias {{.*}} @[[E_VFTABLE]],
// MSABI-DAG: @"??_7F@@6B@" = {{.*}} alias {{.*}} @[[F_VFTABLE]],
