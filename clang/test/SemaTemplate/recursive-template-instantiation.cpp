// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> void f(T* t) { // expected-note{{failed template argument deduction}}
  f(*t); // expected-error{{no matching function}}\
         // expected-note 3{{requested here}}
}

void test_f(int ****p) {
  f(p); // expected-note{{requested here}}
}
