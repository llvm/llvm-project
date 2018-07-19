// RUN: %clang_cc1 -fms-extensions -fno-rtti -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=19.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2015
// RUN: %clang_cc1 -fms-extensions -fno-rtti -emit-llvm %s -o - -triple=i386-pc-win32 -fms-compatibility-version=18.00 | FileCheck %s --check-prefix=CHECK --check-prefix=MSVC2013

// CHECK: @"\01??_7B@?1??foo@A@@QAEXH@Z@6B@" =
// CHECK: @"\01??_7D@C@?1??foo@@YAXXZ@6B@" =
// MSVC2013: define {{.*}} @"\01?baz@E@?3??bar@C@?1??foo@@YAXXZ@QAEXXZ@QAEXXZ"(
// MSVC2015: define {{.*}} @"\01?baz@E@?1??bar@C@?1??foo@@YAXXZ@QAEXXZ@QAEXXZ"(

// Microsoft Visual C++ ABI examples.
struct A {
  void foo (int) {
    struct B { virtual ~B() {} };
    B();
  }
};
inline void foo () {
  struct C {
    struct D { virtual ~D() {} };
    void bar () {
      struct E {
        void baz() { }
      };
      E().baz();
    }
  };
  A().foo(0);
  C::D();
  C().bar();
}
void call () {
  foo();
}
