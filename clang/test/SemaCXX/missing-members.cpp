// RUN: %clang_cc1 -fsyntax-only -verify %s
namespace A {
  namespace B {
    class C { }; // expected-note {{'A::B::C' declared here}}
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
  A::B::D::E; // expected-error-re {{no member named 'D' in namespace 'A::B'{{$}}}}
  // FIXME: The typo corrections below should be suppressed since A::B::C
  // doesn't have a member named D.
  B::B::C::D; // expected-error {{no member named 'C' in 'B::B'; did you mean 'A::B::C'?}} \
              // expected-error-re {{no member named 'D' in 'A::B::C'{{$}}}}
  ::C::D; // expected-error-re {{no member named 'C' in the global namespace{{$}}}}
}

int A::B::i = 10; // expected-error {{no member named 'i' in namespace 'A::B'}}
int A::B::C::i = 10; // expected-error {{no member named 'i' in 'A::B::C'}}
int A::B::S::i = 10; // expected-error {{no member named 'i' in 'A::B::S'}}
int A::B::U::i = 10; // expected-error {{no member named 'i' in 'A::B::U'}}

using A::B::D; // expected-error {{no member named 'D' in namespace 'A::B'}}

struct S : A::B::C { 
  using A::B::C::f; // expected-error {{no member named 'f' in 'A::B::C'}}
  
};

struct S1 {};

struct S2 : S1 {};

struct S3 : S2 {
  void run();
};

struct S4: S3 {};

void test(S4 *ptr) {
  ptr->S1::run();  // expected-error {{no member named 'run' in 'S1'}}
}
