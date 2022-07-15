// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace test0 {
  struct A {
    A() = default;
    int x;
    int y;

    A(const A&) = delete; // expected-note {{'A' has been explicitly marked deleted here}}
  };

  void foo(...);

  void test() {
    A a;
    foo(a); // expected-error {{call to deleted constructor of 'test0::A'}}
  }
}

namespace test1 {
  struct A {
    A() = default;
    int x;
    int y;

  private:
    A(const A&) = default; // expected-note {{declared private here}}
  };

  void foo(...);

  void test() {
    A a;
    foo(a); // expected-error {{calling a private constructor of class 'test1::A'}}
  }
}

// Don't enforce this in an unevaluated context.
namespace test2 {
  struct A {
    A(const A&) = delete; // expected-note {{marked deleted here}}
  };

  typedef char one[1];
  typedef char two[2];

  one &meta(bool);
  two &meta(...);

  void a(A &a) {
    char check[sizeof(meta(a)) == 2 ? 1 : -1];
  }

  void b(A &a) {
    meta(a); // expected-error {{call to deleted constructor}}
  }
}
