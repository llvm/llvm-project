// RUN: %clang_cc1 -fsyntax-only -std=c++03 -verify %s

class string;
void f(const string& s, ...) {  // expected-note {{parameter of type 'const string &' is declared here}}
  __builtin_va_list ap;
  __builtin_va_start(ap, s); // expected-warning {{passing an object of reference type to 'va_start' has undefined behavior}}
}

void g(register int i, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, i); // okay
}
