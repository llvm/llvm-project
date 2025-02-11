// RUN: %clang_cc1 -std=c17 -DPRE_C23 -fsyntax-only -verify -isystem %S/Inputs %s
// RUN: %clang_cc1 -std=c23 -DPOST_C23 -fsyntax-only -verify -isystem %S/Inputs %s

#include <stdarg.h>
#include <stddef.h>

int printf(const char *restrict, ...);

#if PRE_C23
void test_pre_c23(unsigned char x) {
  printf("%lb %lB", (long) 10, (long) 10); // expected-warning{{invalid conversion specifier 'b'}}
                                           // expected-warning@-1{{invalid conversion specifier 'B'}}
                                           // expected-warning@-2{{data argument not used by format string}}
  printf("%llb %llB", (long long) 10, (long long) 10); // expected-warning{{invalid conversion specifier 'b'}}
                                                       // expected-warning@-1{{invalid conversion specifier 'B'}}
                                                       // expected-warning@-2{{data argument not used by format string}}
  printf("%0b%0B", -1u, -1u); // expected-warning{{invalid conversion specifier 'b'}}
                              // expected-warning@-1{{invalid conversion specifier 'B'}}
                              // expected-warning@-2{{data argument not used by format string}}
  printf("%#b %#15.8B\n", 10, 10u); // expected-warning{{invalid conversion specifier 'b'}}
                                    // expected-warning@-1{{invalid conversion specifier 'B'}}
                                    // expected-warning@-2{{data argument not used by format string}}
  printf("%'b\n", 123456789); // expected-warning{{invalid conversion specifier 'b'}}
  printf("%'B\n", 123456789); // expected-warning{{invalid conversion specifier 'B'}}
  printf("%hhb %hhB", x, x); // expected-warning{{invalid conversion specifier 'b'}}
                             // expected-warning@-1{{invalid conversion specifier 'B'}}
                             // expected-warning@-2{{data argument not used by format string}}
}
#endif

#if POST_C23
void test_post_c23(unsigned char x) {
  printf("%lb %lB", (long) 10, (long) 10); // no-warning
  printf("%llb %llB", (long long) 10, (long long) 10); // no-warning
  printf("%0b%0B", -1u, -1u); // no-warning
  printf("%#b %#15.8B\n", 10, 10u); // no-warning
  printf("%'b\n", 123456789); // expected-warning{{results in undefined behavior with 'b' conversion specifier}}
  printf("%'B\n", 123456789); // expected-warning{{results in undefined behavior with 'B' conversion specifier}}
  printf("%hhb %hhB", x, x); // no-warning
}
#endif
