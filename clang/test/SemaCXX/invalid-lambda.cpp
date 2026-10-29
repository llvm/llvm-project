// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++26 %s

decltype([]()->decltype(this) { }) a; // expected-error {{invalid use of 'this' outside of a non-static member function}}

#if __cplusplus >= 202302L
namespace GH175173 {

struct A {
  void f(this A&) {
    auto l = [](auto g, decltype(g(this))){};
    // expected-error@-1 {{invalid use of 'this'}}
  }
};

struct B {
  void f() {
    auto l = [](int g, decltype(g(this))){};
    // expected-error@-1 {{called object type 'int'}}
  }
};

}
#endif

#if __cplusplus == 201103L

namespace GH150733 {
auto L = [](auto a) {
  // expected-error@-1 {{'auto' not allowed in lambda}}
  auto M = [](decltype(a)) -> {
    // expected-error@-1 {{expected a type}}
  };
};
}

#endif

namespace GH170185 {

struct Foo; // expected-note 2{{forward declaration}}

template <Foo X>
// expected-error@-1 {{non-type template parameter has incomplete type}}
struct Arr;

Foo f; // expected-error {{variable has incomplete type}}

void foo() {
    (void)[]() {
        (void)[&](Arr<f>) {

        };
    }();
}

}
