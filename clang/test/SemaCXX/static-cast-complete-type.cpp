// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T> struct S {
  S(int);
};

struct T; // expected-note{{forward declaration of 'T'}}

void f() {
  S<int> s0 = static_cast<S<int> >(0);
  S<void*> s1 = static_cast<S<void*> >(00);

  (void)static_cast<T>(10); // expected-error{{'T' is an incomplete type}}
}
