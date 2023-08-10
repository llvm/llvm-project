// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -Wno-bitfield-constant-conversion -verify %s
// RUN: %clang_cc1 -verify=ref -Wno-bitfield-constant-conversion %s

// expected-no-diagnostics
// ref-no-diagnostics

namespace Basic {
  struct A {
    unsigned int a : 2;
    constexpr A() : a(0) {}
    constexpr A(int a) : a(a) {}
  };

  constexpr A a{1};
  static_assert(a.a == 1, "");

  constexpr A a2{10};
  static_assert(a2.a == 2, "");


  constexpr int storeA() {
    A a;
    a.a = 10;

    return a.a;
  }
  static_assert(storeA() == 2, "");

  constexpr int storeA2() {
    A a;
    return a.a = 10;
  }
  static_assert(storeA2() == 2, "");

  // TODO: +=, -=, etc. operators.
}

namespace Overflow {
  struct A {int c:3;};

  constexpr int f() {
    A a1{3};
    return a1.c++;
  }

  static_assert(f() == 3, "");
}
