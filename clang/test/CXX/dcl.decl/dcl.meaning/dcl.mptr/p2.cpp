// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct A0 {
  struct B0;

  template<typename U>
  struct C0 {
    struct D0;

    template<typename V>
    struct E0;
  };
};

template<typename T>
int A0<T>::B0::* f0();

template<typename T>
int A0<T>::B1::* f1();

template<typename T>
int A0<T>::C0<int>::* f2(); // expected-error {{expected unqualified-id}}

template<typename T>
int A0<T>::C1<int>::* f3(); // expected-error {{no member named 'C1' in 'A0<T>'}}
                            // expected-error@-1 {{expected ';' after top level declarator}}

template<typename T>
int A0<T>::template C2<int>::* f4();

template<typename T>
int A0<T>::template C0<int>::D0::* f5();

template<typename T>
int A0<T>::template C2<int>::D1::* f6();

template<typename T>
int A0<T>::template C0<int>::E0<int>::* f7(); // expected-error {{use 'template' keyword to treat 'E0' as a dependent template name}}
                                              // expected-error@-1 {{expected unqualified-id}}

template<typename T>
int A0<T>::template C2<int>::E1<int>::* f8(); // expected-error {{no member named 'C2' in 'A0<T>'}}

template<typename T>
int A0<T>::template C0<int>::template E0<int>::* f9();

template<typename T>
int A0<T>::template C2<int>::template E1<int>::* f10();

namespace TypoCorrection {
  template<typename T>
  struct A {
    template<typename U>
    struct Typo; // expected-note {{'Typo' declared here}}
  };

  template<typename T>
  int A<T>::template typo<int>::* f();

  template<typename T>
  int A<T>::typo<int>::* g(); // expected-error {{no template named 'typo' in 'A<T>'; did you mean 'Typo'?}}
                              // expected-error@-1 {{expected unqualified-id}}
}
