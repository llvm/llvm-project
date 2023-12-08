// RUN: %clang_cc1 %s -fsyntax-only -verify

struct X {
public __attribute__((unavailable)): // expected-error {{access specifier can only have annotation attributes}}
  void foo();
private __attribute__((annotate("foobar"))):
    void bar();
private __attribute__((annotate())): // expected-error {{'annotate' attribute takes at least 1 argument}}
};

void f(X x) {
  x.foo();
}
