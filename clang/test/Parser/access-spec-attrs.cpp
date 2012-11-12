// RUN: %clang_cc1 %s -fsyntax-only -verify

struct X {
public __attribute__((unavailable)): // expected-error {{access specifier can only have annotation attributes}}
  void foo();
private __attribute__((annotate("foobar"))):
    void bar();
};

void f(X x) {
  x.foo();
}
