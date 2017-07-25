// RUN: %clang_cc1 -triple x86_64-windows-gnu -fdeclspec -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-itanium -fdeclspec -emit-llvm -o - %s | FileCheck %s

struct __declspec(dllexport) A {
  virtual void m();
};
struct __declspec(dllexport) B {
  virtual void m();
};
struct __declspec(dllexport) C : A, B {
  virtual void m();
};
void C::m() {}
// CHECK: define dllexport void @_ZThn8_N1C1mEv

struct Base {
  virtual void m();
};
struct __declspec(dllexport) Derived : virtual Base {
  virtual void m();
};
void Derived::m() {}
// CHECK: define dllexport void @_ZTv0_n24_N7Derived1mEv
