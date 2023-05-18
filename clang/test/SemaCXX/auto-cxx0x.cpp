// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1y
void f() {
  auto int a; // expected-warning {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
  int auto b; // expected-error{{cannot combine with previous 'int' declaration specifier}}
}

typedef auto PR25449(); // expected-error {{'auto' not allowed in typedef}}

thread_local auto x; // expected-error {{requires an initializer}}

void g() {
  [](auto){}(0);
#if __cplusplus == 201103L
  // expected-error@-2 {{'auto' not allowed in lambda parameter}}
#endif
}

void rdar47689465() {
  int x = 0;
  [](auto __attribute__((noderef)) *){}(&x);
#if __cplusplus == 201103L
  // expected-error@-2 {{'auto' not allowed in lambda parameter}}
#endif
}
