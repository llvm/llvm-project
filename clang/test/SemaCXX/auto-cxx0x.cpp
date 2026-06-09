// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1y
void f() {
  auto int a; // expected-error {{'auto' cannot be combined with a type specifier}}
  int auto b; // expected-error {{'auto' cannot be combined with a type specifier}}
  unsigned auto int x; // expected-error {{'auto' cannot be combined with a type specifier}} expected-error-re {{'{{.*}}' cannot be signed or unsigned}}
  signed auto int s; // expected-error {{'auto' cannot be combined with a type specifier}} expected-error-re {{'{{.*}}' cannot be signed or unsigned}}
  auto double y; // expected-error {{'auto' cannot be combined with a type specifier}}
  auto float z; // expected-error {{'auto' cannot be combined with a type specifier}}
  long auto int l; // expected-error {{'auto' cannot be combined with a type specifier}} expected-error-re {{'long {{.*}}' is invalid}}
  auto int arr[10]; // expected-error {{'auto' cannot be combined with a type specifier}}
}

typedef auto PR25449(); // expected-error {{'auto' not allowed in typedef}}

thread_local auto x; // expected-error {{requires an initializer}}

void g() {
  [](auto){}(0);
#if __cplusplus == 201103L
  // expected-error@-2 {{'auto' not allowed in lambda parameter before C++14}}
#endif
}

void rdar47689465() {
  int x = 0;
  [](auto __attribute__((noderef)) *){}(&x);
#if __cplusplus == 201103L
  // expected-error@-2 {{'auto' not allowed in lambda parameter before C++14}}
#endif
}
