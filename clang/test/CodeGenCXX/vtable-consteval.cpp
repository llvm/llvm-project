// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o - -triple x86_64-linux | FileCheck %s --check-prefix=ITANIUM --implicit-check-not=DoNotEmit
// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o - -triple x86_64-windows | FileCheck %s --check-prefix=MSABI --implicit-check-not=DoNotEmit

// FIXME: The MSVC ABI rule in use here was discussed with MS folks prior to
// them implementing virtual consteval functions, but we do not know for sure
// if this is the ABI rule they will use.

// ITANIUM-DAG: @_ZTV1A = {{.*}} constant { [2 x i8*] } {{.*}} null, {{.*}} @_ZTI1A
// MSABI-DAG: @[[A_VFTABLE:.*]] = {{.*}} constant { [1 x i8*] } {{.*}} @"??_R4A@@6B@"
struct A {
  virtual consteval void DoNotEmit_f() {}
};
// ITANIUM-DAG: @a = {{.*}}global { i8** } { {{.*}} @_ZTV1A,
// MSABI-DAG: @"?a@@3UA@@A" = {{.*}}global { i8** } { i8** @"??_7A@@6B@" }
A a;

// ITANIUM-DAG: @_ZTV1B = {{.*}} constant { [4 x i8*] } {{.*}} null, {{.*}} @_ZTI1B {{.*}} @_ZN1B1fEv {{.*}} @_ZN1B1hEv
// MSABI-DAG: @[[B_VFTABLE:.*]] = {{.*}} constant { [3 x i8*] } {{.*}} @"??_R4B@@6B@" {{.*}} @"?f@B@@UEAAXXZ" {{.*}} @"?h@B@@UEAAXXZ"
struct B {
  virtual void f() {}
  virtual consteval void DoNotEmit_g() {}
  virtual void h() {}
};
// ITANIUM-DAG: @b = {{.*}}global { i8** } { {{.*}} @_ZTV1B,
// MSABI-DAG: @"?b@@3UB@@A" = {{.*}}global { i8** } { i8** @"??_7B@@6B@" }
B b;

// ITANIUM-DAG: @_ZTV1C = {{.*}} constant { [4 x i8*] } {{.*}} null, {{.*}} @_ZTI1C {{.*}} @_ZN1CD1Ev {{.*}} @_ZN1CD0Ev
// MSABI-DAG: @[[C_VFTABLE:.*]] = {{.*}} constant { [2 x i8*] } {{.*}} @"??_R4C@@6B@" {{.*}} @"??_GC@@UEAAPEAXI@Z"
struct C {
  virtual ~C() = default;
  virtual consteval C &operator=(const C&) = default;
};
// ITANIUM-DAG: @c = {{.*}}global { i8** } { {{.*}} @_ZTV1C,
// MSABI-DAG: @"?c@@3UC@@A" = {{.*}}global { i8** } { i8** @"??_7C@@6B@" }
C c;

// ITANIUM-DAG: @_ZTV1D = {{.*}} constant { [4 x i8*] } {{.*}} null, {{.*}} @_ZTI1D {{.*}} @_ZN1DD1Ev {{.*}} @_ZN1DD0Ev
// MSABI-DAG: @[[D_VFTABLE:.*]] = {{.*}} constant { [2 x i8*] } {{.*}} @"??_R4D@@6B@" {{.*}} @"??_GD@@UEAAPEAXI@Z"
struct D : C {};
// ITANIUM-DAG: @d = {{.*}}global { i8** } { {{.*}} @_ZTV1D,
// MSABI-DAG: @"?d@@3UD@@A" = {{.*}}global { i8** } { i8** @"??_7D@@6B@" }
D d;

// ITANIUM-DAG: @_ZTV1E = {{.*}} constant { [3 x i8*] } {{.*}} null, {{.*}} @_ZTI1E {{.*}} @_ZN1E1fEv
// MSABI-DAG: @[[E_VFTABLE:.*]] = {{.*}} constant { [2 x i8*] } {{.*}} @"??_R4E@@6B@" {{.*}} @"?f@E@@UEAAXXZ"
struct E { virtual void f() {} };
// ITANIUM-DAG: @e = {{.*}}global { i8** } { {{.*}} @_ZTV1E,
// MSABI-DAG: @"?e@@3UE@@A" = {{.*}}global { i8** } { i8** @"??_7E@@6B@" }
E e;

// ITANIUM-DAG: @_ZTV1F = {{.*}} constant { [3 x i8*] } {{.*}} null, {{.*}} @_ZTI1F {{.*}} @_ZN1E1fEv
// MSABI-DAG: @[[F_VFTABLE:.*]] = {{.*}} constant { [2 x i8*] } {{.*}} @"??_R4F@@6B@" {{.*}} @"?f@E@@UEAAXXZ"
struct F : E { virtual consteval void DoNotEmit_g(); };
// ITANIUM-DAG: @f = {{.*}}global { i8** } { {{.*}} @_ZTV1F,
// MSABI-DAG: @"?f@@3UF@@A" = {{.*}}global { i8** } { i8** @"??_7F@@6B@" }
F f;

// MSABI-DAG: @"??_7A@@6B@" = {{.*}} alias {{.*}} @[[A_VFTABLE]],
// MSABI-DAG: @"??_7B@@6B@" = {{.*}} alias {{.*}} @[[B_VFTABLE]],
// MSABI-DAG: @"??_7C@@6B@" = {{.*}} alias {{.*}} @[[C_VFTABLE]],
// MSABI-DAG: @"??_7D@@6B@" = {{.*}} alias {{.*}} @[[D_VFTABLE]],
// MSABI-DAG: @"??_7E@@6B@" = {{.*}} alias {{.*}} @[[E_VFTABLE]],
// MSABI-DAG: @"??_7F@@6B@" = {{.*}} alias {{.*}} @[[F_VFTABLE]],
