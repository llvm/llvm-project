// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -Wno-bitfield-constant-conversion -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both -Wno-bitfield-constant-conversion %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -Wno-bitfield-constant-conversion -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both -Wno-bitfield-constant-conversion %s

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

#if __cplusplus >= 202002
  struct Init1 {
    unsigned a : 2 = 1;
  };
  constexpr Init1 I1{};
  static_assert(I1.a == 1, "");

  struct Init2 {
    unsigned a : 2 = 100;
  };
  constexpr Init2 I2{};
  static_assert(I2.a == 0, "");
#endif

  struct Init3 {
    unsigned a : 2;
    constexpr Init3() : a(100) {}
  };
  constexpr Init3 I3{};
  static_assert(I3.a == 0, "");
}

namespace Overflow {
  struct A {int c:3;};

  constexpr int f() {
    A a1{3};
    return a1.c++;
  }

  static_assert(f() == 3, "");
}

namespace Compound {
  struct A {
    unsigned int a : 2;
    constexpr A() : a(0) {}
    constexpr A(int a) : a(a) {}
  };

  constexpr unsigned add() {
    A a;
    a.a += 10;
    return a.a;
  }
  static_assert(add() == 2, "");

  constexpr unsigned sub() {
    A a;
    a.a -= 10;
    return a.a;
  }
  static_assert(sub() == 2, "");

  constexpr unsigned mul() {
    A a(1);
    a.a *= 5;
    return a.a;
  }
  static_assert(mul() == 1, "");

  constexpr unsigned div() {
    A a(2);
    a.a /= 2;
    return a.a;
  }
  static_assert(div() == 1, "");
}

namespace test0 {
  extern int int_source();
  struct A {
    int aField;
    int bField;
  };

  struct B {
    int onebit : 2;
    int twobit : 6;
    int intField;
  };

  struct C : A, B {
  };

  void b(C &c) {
    c.onebit = int_source();
  }
}

namespace NonConstBitWidth {
  int n3 = 37; // both-note {{declared here}}
  struct S {
    int l : n3; // both-error {{constant expression}} \
                // both-note {{read of non-const variable}}
  };
}

namespace IncDecOverflow {
  constexpr bool test1() {
    struct {unsigned u: 5; } a {};
    a.u--;
    return a.u == 31;
  }
  static_assert(test1(), "");

  constexpr bool test2() {
    struct {unsigned u: 5; } a {};
    --a.u;
    return a.u == 31;
  }
  static_assert(test2(), "");

  constexpr bool test3() {
    int x = 0;
    struct {unsigned u: 5; } a {};
    x = a.u--;
    return a.u == 31;
  }
  static_assert(test3(), "");

  constexpr bool test4() {
    int x = 0;
    struct {unsigned u: 5; } a {};
    x = --a.u;
    return a.u == 31;
  }
  static_assert(test4(), "");

  constexpr bool test5() {
    struct {unsigned u: 5; } a {};
    a.u = 31;
    ++a.u;

    return a.u == 0;
  }
  static_assert(test5(), "");

  constexpr bool test6() {
    struct {unsigned u: 5; } a {};
    a.u = 31;
    ++a.u;

    return a.u == 0;
  }
  static_assert(test6(), "");

  constexpr bool test7() {
    struct {unsigned u: 5; } a {};
    a.u = 31;
    a.u++;

    return a.u == 0;
  }
  static_assert(test7(), "");

  constexpr bool test8() {
    int x = 0;
    struct {unsigned u: 5; } a {};
    a.u = 31;
    x = a.u++;

    return a.u == 0;
  }
  static_assert(test8(), "");

  constexpr bool test9() {
    int x = 0;
    struct {unsigned u: 5; } a {};
    a.u = 31;
    x = ++a.u;

    return a.u == 0;
  }
  static_assert(test9(), "");
}
