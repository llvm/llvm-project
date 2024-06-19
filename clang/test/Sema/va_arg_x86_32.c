// RUN: %clang_cc1 -fsyntax-only -verify -triple=i686-pc-linux-gnu %s

int a(void) {
  __builtin_va_arg((char*)0, int); // expected-error {{expression is not assignable}}
  // expected-note@-1 {{add '*' to dereference it}}
  __builtin_va_arg((void*){0}, int); // expected-error {{first argument to 'va_arg' is of type 'void *'}}
}
