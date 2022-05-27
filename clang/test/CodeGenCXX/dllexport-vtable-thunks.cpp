// RUN: %clang_cc1 -triple x86_64-windows-gnu     -fdeclspec -emit-llvm -o - %s | FileCheck %s -DDSO_ATTRS="dso_local dllexport"
// RUN: %clang_cc1 -triple x86_64-windows-itanium -fdeclspec -emit-llvm -o - %s | FileCheck %s -DDSO_ATTRS="dso_local dllexport"
// RUN: %clang_cc1 -triple x86_64-scei-ps4        -fdeclspec -emit-llvm -o - %s | FileCheck %s -DDSO_ATTRS=dllexport
// RUN: %clang_cc1 -triple x86_64-sie-ps5         -fdeclspec -emit-llvm -o - %s | FileCheck %s -DDSO_ATTRS=dllexport

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
// CHECK: define{{.*}} [[DSO_ATTRS]] void @_ZThn8_N1C1mEv

struct Base {
  virtual void m();
};
struct __declspec(dllexport) Derived : virtual Base {
  virtual void m();
};
void Derived::m() {}
// CHECK: define{{.*}} [[DSO_ATTRS]] void @_ZTv0_n24_N7Derived1mEv
