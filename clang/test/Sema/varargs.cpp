// RUN: %clang_cc1 -fsyntax-only -verify %s

class string;
void f(const string& s, ...) {  // expected-note {{parameter of type 'const string &' is declared here}}
  __builtin_va_list ap;
  __builtin_va_start(ap, s); // expected-warning {{'va_start' has undefined behavior with reference types}}
}
