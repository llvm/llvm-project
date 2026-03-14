// RUN: %clang_cc1 -fms-compatibility -fdelayed-template-parsing -fsyntax-only -verify %s

template <typename T>
class A {
};
typedef int TInt;

template class A<int>;  // expected-note {{previous explicit instantiation is here}}
template class A<TInt>; // expected-warning {{duplicate explicit instantiation of 'A<int>' ignored as a Microsoft extension}}
