// No PCH:
// RUN: %clang_cc1 -pedantic -std=c++1y -include %s -verify %s
//
// With PCH:
// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t -verify %s

// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t -verify %s

#ifndef HEADER
#define HEADER

auto counter = [a(0)] () mutable { return a++; };
int x = counter();

template<typename T> void f(T t) {
  [t(t)] { int n = t; } ();
}

#else

int y = counter();

void g() {
  f(0); // ok
  // expected-error@18 {{lvalue of type 'const char *const'}}
  // expected-note@18 {{substituting into a lambda}}
  f("foo"); // expected-note {{here}}
}

#endif
