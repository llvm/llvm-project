// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

void f() {
  auto a = f(); // expected-error {{variable has incomplete type 'void'}}
  auto &b = f(); // expected-error {{cannot form a reference to 'void'}}
  auto *c = f(); // expected-error {{incompatible initializer of type 'void'}}

  auto d(f()); // expected-error {{variable has incomplete type 'void'}}
  auto &&e(f()); // expected-error {{cannot form a reference to 'void'}}
  auto *g(f()); // expected-error {{incompatible initializer of type 'void'}}

  (void)new auto(f()); // expected-error {{allocation of incomplete type 'void'}}
  (void)new auto&(f()); // expected-error {{cannot form a reference to 'void'}}
  (void)new auto*(f()); // expected-error {{incompatible constructor argument of type 'void'}}
}
