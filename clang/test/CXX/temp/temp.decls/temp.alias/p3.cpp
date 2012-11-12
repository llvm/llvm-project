// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// The example given in the standard (this is rejected for other reasons anyway).
template<class T> struct A;
template<class T> using B = typename A<T>::U; // expected-error {{no type named 'U' in 'A<T>'}}
template<class T> struct A {
  typedef B<T> U; // expected-note {{in instantiation of template type alias 'B' requested here}}
};
B<short> b;

template<typename T> using U = int;
// FIXME: This is illegal, but probably only because CWG1044 missed this paragraph.
template<typename T> using U = U<T>;
