// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -fixit -x c %t
// RUN: %clang_cc1 -pedantic -Werror -Wno-invalid-noreturn -x c %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

struct s; // expected-note{{previous use is here}}

union s *s1; // expected-error{{use of 's' with tag type that does not match previous declaration}}

struct Point {
  float x, y, z;
};

struct Point *get_origin();

void test_point() {
  (void)get_origin->x; // expected-error {{base of member reference is a function; perhaps you meant to call it with no arguments?}}
}

// These errors require C11.
#if __STDC_VERSION__ > 199901L
void noreturn_1() _Noreturn; // expected-error {{must precede function declarator}}
void noreturn_1() {
  return; // expected-warning {{should not return}}
}
void noreturn_2() _Noreturn { // expected-error {{must precede function declarator}}
  return; // expected-warning {{should not return}}
}
#endif
