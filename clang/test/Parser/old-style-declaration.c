// RUN: %clang_cc1 -fsyntax-only -verify -Wold-style-declaration %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wextra %s

static int x0;
int __attribute__ ((aligned (16))) static x1; // expected-warning {{'static' is not at beginning of declaration}}

extern int x2;
int extern x3; // expected-warning {{'extern' is not at beginning of declaration}}

typedef int x4;
int typedef x5; // expected-warning {{'typedef' is not at beginning of declaration}}

void g (int);

void
f (void)
{
  auto int x6 = 0;
  int auto x7 = 0; // expected-warning {{'auto' is not at beginning of declaration}}
  register int x8 = 0;
  int register x9 = 0; // expected-warning {{'register' is not at beginning of declaration}}
  g (x6 + x7 + x8 + x9);
}

const static int x10; // expected-warning {{'static' is not at beginning of declaration}}

/* Attributes are OK before storage class specifiers, since some
   attributes are like such specifiers themselves.  */

__attribute__((format(printf, 1, 2))) static void h (const char *, ...);
__attribute__((format(printf, 1, 2))) void static i (const char *, ...); // expected-warning {{'static' is not at beginning of declaration}}

static __thread int var = 5; // not-expected-warning {{'__thread' is not at beginning of declaration}}