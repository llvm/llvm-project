// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic
// RUN: %clang_cc1 %s -verify -fsyntax-only -x c++ -pedantic

// This file contains UTF-8; please do not fix!


extern void \u00FCber(int);
extern void \U000000FCber(int); // redeclaration, no warning
#ifdef __cplusplus
// expected-note@-2 + {{candidate function not viable}}
#else
// expected-note@-4 + {{declared here}}
#endif

void goodCalls(void) {
  \u00FCber(0);
  \u00fcber(1);
  über(2);
  \U000000FCber(3);
  \u{FC}ber(4); // expected-warning {{Clang extension}}
  \N{LATIN SMALL LETTER U WITH DIAERESIS}ber(4); // expected-warning {{Clang extension}}
}

void badCalls(void) {
  \u00FCber(0.5); // expected-warning{{implicit conversion from 'double' to 'int'}}
  \u00fcber = 0; // expected-error{{non-object type 'void (int)' is not assignable}}

  über(1, 2);
  \U000000FCber();
#ifdef __cplusplus
  // expected-error@-3 {{no matching function}}
  // expected-error@-3 {{no matching function}}
#else
  // expected-error@-6 {{too many arguments to function call, expected 1, have 2}}
  // expected-error@-6 {{too few arguments to function call, expected 1, have 0}}
#endif
}
