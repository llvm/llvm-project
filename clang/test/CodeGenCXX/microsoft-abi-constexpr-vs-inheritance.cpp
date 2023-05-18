// RUN: %clang_cc1 -std=c++11 -fno-rtti -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

struct A {
  constexpr A(int x) : x(x) {}
  virtual void f();
  int x;
};

A a(42);
// CHECK: @"?a@@3UA@@A" = dso_local global %struct.A { ptr @"??_7A@@6B@", i32 42 }, align 4

struct B {
  constexpr B(int y) : y(y) {}
  virtual void g();
  int y;
};

struct C : A, B {
  constexpr C() : A(777), B(13) {}
};

C c;
// CHECK: @"?c@@3UC@@A" = dso_local global { ptr, i32, ptr, i32 } { ptr @"??_7C@@6BA@@@", i32 777, ptr @"??_7C@@6BB@@@", i32 13 }
