// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Instantiating a special member function that is defaulted outside its class
// template, while a non-default floating-point pragma is in effect at the
// point of instantiation, used to trigger an assertion in
// FunctionDecl::setDefaultedOrDeletedInfo.

template <class T> struct S {
  S &operator=(const S &);
  S &operator=(S &&);
};
template <class T> S<T> &S<T>::operator=(const S &) = default;
template <class T> S<T> &S<T>::operator=(S &&) = default;

#pragma clang fp contract(off)
void f(S<int> &a, const S<int> &b) {
  a = b;
  a = static_cast<S<int> &&>(a);
}
