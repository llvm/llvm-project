// RUN: %clang_cc1 -fsyntax-only -verify %s
namespace A {
  namespace B {
    class C { };
    struct S { };
    union U { };
  }
}

void f() {
  A::B::i; // expected-error {{no member named 'i' in namespace 'A::B'}}
  A::B::C::i; // expected-error {{no member named 'i' in 'A::B::C'}}
  ::i; // expected-error {{no member named 'i' in the global namespace}}
}

namespace B {
  class B { };
}

void g() {
  A::B::D::E; // expected-error {{no member named 'D' in namespace 'A::B'}}
  B::B::C::D; // expected-error {{no member named 'C' in 'B::B'}}
  ::C::D; // expected-error {{no member named 'C' in the global namespace}}
}

int A::B::i = 10; // expected-error {{no member named 'i' in namespace 'A::B'}}
int A::B::C::i = 10; // expected-error {{no member named 'i' in 'A::B::C'}}
int A::B::S::i = 10; // expected-error {{no member named 'i' in 'A::B::S'}}
int A::B::U::i = 10; // expected-error {{no member named 'i' in 'A::B::U'}}

using A::B::D; // expected-error {{no member named 'D' in namespace 'A::B'}}

struct S : A::B::C { 
  using A::B::C::f; // expected-error {{no member named 'f' in 'A::B::C'}}
  
};
