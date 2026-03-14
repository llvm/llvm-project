// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR7837

template<class T> struct C1 { void operator()(T); };
template<class T> struct C2; // expected-note {{template is declared here}}
template<class T> void foo(T);
void wrap() {
  foo(&C1<int>::operator());
  foo(&C1<int>::operator+); // expected-error {{no member named 'operator+' in 'C1<int>'}}
  foo(&C2<int>::operator+); // expected-error {{implicit instantiation of undefined template 'C2<int>'}}
}
