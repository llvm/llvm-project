// RUN: %clang_cc1 -std=c17 -fsyntax-only -verify=c17 -isystem %S/Inputs %s
// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify=c23 -isystem %S/Inputs %s

#include <stdarg.h>
#include <stddef.h>

int printf(const char *restrict, ...);

void test(unsigned char x) {
  printf("%lb %lB", (long) 10, (long) 10); // c17-warning{{conversion specifier 'b' requires a C standard library compatible with C23; data argument may not be used by format}}
                                           // c17-warning@-1{{conversion specifier 'B' requires a C standard library compatible with C23; data argument may not be used by format}}

  printf("%llb %llB", (long long) 10, (long long) 10); // c17-warning{{conversion specifier 'b' requires a C standard library compatible with C23; data argument may not be used by format}}
                                                       // c17-warning@-1{{conversion specifier 'B' requires a C standard library compatible with C23; data argument may not be used by format}}

  printf("%0b%0B", -1u, -1u); // c17-warning{{conversion specifier 'b' requires a C standard library compatible with C23; data argument may not be used by format}}
                              // c17-warning@-1{{conversion specifier 'B' requires a C standard library compatible with C23; data argument may not be used by format}}

  printf("%#b %#15.8B\n", 10, 10u); // c17-warning{{conversion specifier 'b' requires a C standard library compatible with C23; data argument may not be used by format}}
                                    // c17-warning@-1{{conversion specifier 'B' requires a C standard library compatible with C23; data argument may not be used by format}}

  printf("%'b\n", 123456789); // c17-warning{{conversion specifier 'b' requires a C standard library compatible with C23; data argument may not be used by format}}
                              // c17-warning@-1{{results in undefined behavior with 'b' conversion specifier}}
                              // c23-warning@-2{{results in undefined behavior with 'b' conversion specifier}}

  printf("%'B\n", 123456789); // c17-warning{{conversion specifier 'B' requires a C standard library compatible with C23; data argument may not be used by format}}
                              // c17-warning@-1{{results in undefined behavior with 'B' conversion specifier}}
                              // c23-warning@-2{{results in undefined behavior with 'B' conversion specifier}}

  printf("%hhb %hhB", x, x); // c17-warning{{conversion specifier 'b' requires a C standard library compatible with C23; data argument may not be used by format}}
                             // c17-warning@-1{{conversion specifier 'B' requires a C standard library compatible with C23; data argument may not be used by format}}
}
