// RUN: %clang_cc1 -fsyntax-only -verify %s

// There should be no extra errors about missing 'template' keywords.
struct B {
  template <typename T>
  int f(){};
} builder;                // expected-note 2{{'builder' declared here}}

auto a = bilder.f<int>(); // expected-error{{undeclared identifier 'bilder'; did you mean}}
auto b = (*(&bilder+0)).f<int>(); // expected-error{{undeclared identifier 'bilder'; did you mean}}

struct X {
    struct type {};
};

namespace PR48339 {
  struct S {
    template <typename T> static void g(typename T::type) {} // expected-note {{couldn't infer template argument 'T'}}
    template <typename T> void f() { g(typename T::type{}); } // expected-error {{no matching function for call to 'g'}}
  };

  void f() { S{}.f<X>(); } // expected-note {{in instantiation of}}
}
