// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X0 {
  void f(T&);
  
  struct Inner;
  
  static T static_var;
};

template<typename T>
void X0<T>::f(T& t) { 
  t = 1; // expected-error{{incompatible type}}
}

template<typename T>
struct X0<T>::Inner {
  T member;
};

template<typename T>
T X0<T>::static_var = 1; // expected-error{{cannot initialize}}

extern template struct X0<void*>;
template struct X0<void*>; // expected-note 2{{instantiation}}

template struct X0<int>; // expected-note 4{{explicit instantiation definition is here}}

extern template void X0<int>::f(int&); // expected-error{{follows explicit instantiation definition}}
extern template struct X0<int>::Inner; // expected-error{{follows explicit instantiation definition}}
extern template int X0<int>::static_var; // expected-error{{follows explicit instantiation definition}}
extern template struct X0<int>; // expected-error{{follows explicit instantiation definition}}
