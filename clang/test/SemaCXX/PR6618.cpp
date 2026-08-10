// RUN: %clang_cc1 -fsyntax-only -verify %s
class bar; // expected-note {{forward declaration of 'bar'}}
struct zed {
  bar g; // expected-error {{field has incomplete type}}
};
class baz {
  zed h;
};
void f() {
  enum {
    e = sizeof(baz)
  };
}
