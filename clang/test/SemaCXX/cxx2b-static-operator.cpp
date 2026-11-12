// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

// expected-no-diagnostics

namespace A {

struct Foo {
  static int operator()(int a, int b) { return a + b; }
  static int operator[](int a, int b) { return a + b; }
};

void ok() {
  // Should pass regardless of const / volatile
  Foo foo;
  foo(1, 2);
  foo[1, 2];

  const Foo fooC;
  fooC(1, 2);
  fooC[1, 2];

  const Foo fooV;
  fooV(1, 2);
  fooV[1, 2];

  const volatile Foo fooCV;
  fooCV(1, 2);
  fooCV[1, 2];
}

}
