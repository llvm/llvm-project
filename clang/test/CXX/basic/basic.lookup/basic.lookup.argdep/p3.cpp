// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: embellish

namespace test0 {
  namespace A {
    class Foo {
    };

    void foo(const Foo &foo);
  }

  class Test {
    enum E { foo = 0 };

    void test() {
      foo(A::Foo()); // expected-error {{not a function}}
    }
  };
}
