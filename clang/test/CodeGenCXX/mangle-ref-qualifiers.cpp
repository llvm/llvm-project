// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
struct X {
  int f() &;
  int g() &&;
  int h() const &&;
};

// CHECK: define i32 @_ZNR1X1fEv
int X::f() & { return 0; }
// CHECK: define i32 @_ZNO1X1gEv
int X::g() && { return 0; }
// CHECK: define i32 @_ZNKO1X1hEv
int X::h() const && { return 0; }

// CHECK: define void @_Z1fM1XFivREMS_FivOEMS_KFivOE
void f(int (X::*)() &, int (X::*)() &&, int (X::*)() const&&) { }

// CHECK: define void @_Z1g1AIFivEES_IFivREES_IFivOEES_IKFivEES_IKFivREES_IKFivOEES_IVKFivEES_IVKFivREES_IVKFivOEE()
template <class T> struct A {};
void g(A<int()>, A<int()&>, A<int()&&>,
       A<int() const>, A<int() const &>, A<int() const &&>,
       A<int() const volatile>, A<int() const volatile &>, A<int() const volatile &&>) {}
