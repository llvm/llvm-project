// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(void*, int); // expected-note{{candidate function}}
template<typename T>
  void f(T*, long); // expected-note{{candidate function}}

void test_f(int *ip, int i) {
  f(ip, i); // expected-error{{ambiguous}}
}
