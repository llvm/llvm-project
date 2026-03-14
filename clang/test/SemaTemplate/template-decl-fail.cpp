// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> typedef T X; // expected-error{{typedef cannot be a template}}

template<typename T>
enum t0 { A = T::x }; // expected-error{{enumeration cannot be a template}} \
                      // expected-error{{declaration does not declare anything}}

enum e0 {};
template<int x> enum e0 f0(int a=x) {}
