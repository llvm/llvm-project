// RUN: %clang_cc1 -fexceptions -fcxx-exceptions %s -triple=i686-windows-msvc -emit-llvm -o - | FileCheck %s

// When generating thunks using musttail due to inalloca parameters, don't push
// and pop terminate scopes. PR44987

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &o);
  ~NonTrivial();
  int x;
};
struct A {
  virtual void f(NonTrivial o) noexcept;
};
struct B {
  virtual void f(NonTrivial o) noexcept;
};
class C : A, B {
  virtual void f(NonTrivial o) noexcept;
};
C c;

// CHECK-LABEL: define linkonce_odr dso_local x86_thiscallcc void @"?f@C@@G3AEXUNonTrivial@@@Z"(%class.C* %this, <{ %struct.NonTrivial }>* inalloca %0)
// CHECK-NOT: invoke
// CHECK: musttail call x86_thiscallcc void @"?f@C@@EAEXUNonTrivial@@@Z"(%class.C* %{{.*}}, <{ %struct.NonTrivial }>* inalloca %0)
// CHECK-NEXT:  ret void

