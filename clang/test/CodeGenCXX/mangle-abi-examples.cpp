// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s

// CHECK: @_ZTVZN1A3fooEiE1B =
// CHECK: @_ZTVZ3foovEN1C1DE =
// CHECK: define {{.*}} @_ZZZ3foovEN1C3barEvEN1E3bazEv(

// Itanium C++ ABI examples.
struct A {
  void foo (int) {
    struct B { virtual ~B() {} };
    B();
  }
};
void foo () {
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
