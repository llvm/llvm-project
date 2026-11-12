// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

void f();  // expected-note{{possible target for call}}
void f(int);  // expected-note{{possible target for call}}
decltype(f) a;  // expected-error{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}} expected-error {{variable has incomplete type 'decltype(f())' (aka 'void')}}

template<typename T> struct S {
  decltype(T::f) * f; // expected-error {{call to non-static member function without an object argument}}
};

struct K { 
  void f();
  void f(int);
};
S<K> b; // expected-note{{in instantiation of template class 'S<K>' requested here}}

namespace PR13978 {
  template<typename T> struct S { decltype(1) f(); };
  template<typename T> decltype(1) S<T>::f() { return 1; }

  // This case is ill-formed (no diagnostic required) because the decltype
  // expressions are functionally equivalent but not equivalent. It would
  // be acceptable for us to reject this case.
  template<typename T> struct U { struct A {}; decltype(A{}) f(); };
  template<typename T> decltype(typename U<T>::A{}) U<T>::f() {}

  // This case is valid.
  template<typename T> struct V { struct A {}; decltype(typename V<T>::A{}) f(); };
  template<typename T> decltype(typename V<T>::A{}) V<T>::f() {}
}
