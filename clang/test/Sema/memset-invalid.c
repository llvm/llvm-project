// RUN: %clang_cc1 -fsyntax-only %s -verify

char memset(void); // expected-warning {{incompatible redeclaration of library function 'memset'}} expected-note{{'memset' is a builtin with type}}
char test(void) {
  return memset();
}
