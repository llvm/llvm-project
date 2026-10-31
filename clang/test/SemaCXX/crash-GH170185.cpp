// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

struct Foo; // expected-note 2 {{forward declaration of 'Foo'}}

template <Foo X> // expected-error {{non-type template parameter has incomplete type 'Foo'}}
struct Arr;

Foo f; // expected-error {{variable has incomplete type 'Foo'}}

void foo() {
  []() {
    (void)[&](Arr<f>) {};
  }();
}
