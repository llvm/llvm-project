// RUN: %clang_cc1 -fsyntax-only -verify %s

// rdar://problem/8540720
namespace test0 {
  void foo() {
    void bar();
    class A {
      friend void bar();
    };
  }
}

namespace test1 {
  void foo() {
    class A {
      friend void bar(); // expected-error {{no matching function found in local scope}}
    };
  }
}
