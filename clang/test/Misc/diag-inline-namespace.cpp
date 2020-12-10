// RUN: %clang_cc1 -verify %s -std=c++20

// We avoid printing the name of an inline namespace unless it's necessary to
// uniquely identify the target.
namespace N {
  inline namespace A {
    inline namespace B {
      inline namespace C {
        int f, g, h, i, j;
        struct f; struct g; struct h; struct i; struct j;
      }
      struct g;
      struct j;
    }
    struct h;
  }
  struct i;
  struct j;

  template<int*> struct Q; // expected-note 5{{here}}
  Q<&A::B::C::f> q1; // expected-error {{implicit instantiation of undefined template 'N::Q<&N::f>'}}
  Q<&A::B::C::g> q2; // expected-error {{implicit instantiation of undefined template 'N::Q<&N::C::g>'}}
  Q<&A::B::C::h> q3; // expected-error {{implicit instantiation of undefined template 'N::Q<&N::B::h>'}}
  Q<&A::B::C::i> q4; // expected-error {{implicit instantiation of undefined template 'N::Q<&N::A::i>'}}
  Q<&A::B::C::j> q5; // expected-error {{implicit instantiation of undefined template 'N::Q<&N::C::j>'}}

  template<typename> struct R; // expected-note 5{{here}}
  R<struct A::B::C::f> r1; // expected-error {{implicit instantiation of undefined template 'N::R<N::f>'}}
  R<struct A::B::C::g> r2; // expected-error {{implicit instantiation of undefined template 'N::R<N::C::g>'}}
  R<struct A::B::C::h> r3; // expected-error {{implicit instantiation of undefined template 'N::R<N::B::h>'}}
  R<struct A::B::C::i> r4; // expected-error {{implicit instantiation of undefined template 'N::R<N::A::i>'}}
  R<struct A::B::C::j> r5; // expected-error {{implicit instantiation of undefined template 'N::R<N::C::j>'}}

  // Make the name N::C ambiguous.
  inline namespace A { int C; }

  template<int*> struct S; // expected-note 5{{here}}
  S<&A::B::C::f> s1; // expected-error {{implicit instantiation of undefined template 'N::S<&N::f>'}}
  S<&A::B::C::g> s2; // expected-error {{implicit instantiation of undefined template 'N::S<&N::B::C::g>'}}
  S<&A::B::C::h> s3; // expected-error {{implicit instantiation of undefined template 'N::S<&N::B::h>'}}
  S<&A::B::C::i> s4; // expected-error {{implicit instantiation of undefined template 'N::S<&N::A::i>'}}
  S<&A::B::C::j> s5; // expected-error {{implicit instantiation of undefined template 'N::S<&N::B::C::j>'}}

  template<typename> struct T; // expected-note 5{{here}}
  T<struct A::B::C::f> t1; // expected-error {{implicit instantiation of undefined template 'N::T<N::f>'}}
  T<struct A::B::C::g> t2; // expected-error {{implicit instantiation of undefined template 'N::T<N::B::C::g>'}}
  T<struct A::B::C::h> t3; // expected-error {{implicit instantiation of undefined template 'N::T<N::B::h>'}}
  T<struct A::B::C::i> t4; // expected-error {{implicit instantiation of undefined template 'N::T<N::A::i>'}}
  T<struct A::B::C::j> t5; // expected-error {{implicit instantiation of undefined template 'N::T<N::B::C::j>'}}
}
