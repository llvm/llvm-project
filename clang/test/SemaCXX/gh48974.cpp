// RUN: %clang_cc1 -Werror=unused-parameter -Wfatal-errors -verify %s

void a(int &&s) {} // expected-error{{unused parameter 's'}}

void b() {
  int sum = a(0);
}
