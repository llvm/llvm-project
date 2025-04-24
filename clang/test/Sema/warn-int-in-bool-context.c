// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wint-in-bool-context %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c -std=c23 -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wint-in-bool-context %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wall %s

#define ONE 1
#define TWO 2

#define SHIFT(l, r) l << r
#define MM a << a
#define AF 1 << 7

#ifdef __cplusplus
typedef bool boolean;
#else
typedef _Bool boolean;
#endif

enum num {
  zero,
  one,
  two,
};

int test(int a, unsigned b, enum num n) {
  boolean r;
  r = a << a;    // expected-warning {{converting the result of '<<' to a boolean}}
  r = MM;        // expected-warning {{converting the result of '<<' to a boolean}}
  r = (1 << 7);  // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 2UL << 2;  // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 0 << a;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to false}}
  r = 0 << 2;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to false}}
  r = 1 << 0;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 1 << 2;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 1ULL << 2; // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 2 << b;    // expected-warning {{converting the result of '<<' to a boolean}}
  r = (unsigned)(2 << b);
  r = b << 7;
  r = (1 << a); // expected-warning {{converting the result of '<<' to a boolean}}
  r = TWO << a; // expected-warning {{converting the result of '<<' to a boolean}}
  r = a << 7;   // expected-warning {{converting the result of '<<' to a boolean}}
  r = ONE << a; // expected-warning {{converting the result of '<<' to a boolean}}
  if (TWO << a) // expected-warning {{converting the result of '<<' to a boolean}}
    return a;

  a = 1 << 2 ? 0: 1; // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  a = 1 << a ? 0: 1; // expected-warning {{converting the result of '<<' to a boolean}}

  for (a = 0; 1 << a; a++) // expected-warning {{converting the result of '<<' to a boolean}}
    ;

  if (a << TWO) // expected-warning {{converting the result of '<<' to a boolean}}
    return a;

  if (n || two)
    // expected-warning@-1 {{converting the enum constant to a boolean}}
    return a;

  if (n == one || two)
    // expected-warning@-1 {{converting the enum constant to a boolean}}
    return a;

  if (r && two)
    // expected-warning@-1 {{converting the enum constant to a boolean}}
    return a;

  if (two && r)
    // expected-warning@-1 {{converting the enum constant to a boolean}}
    return a;

  if (n == one && two)
    // expected-warning@-1 {{converting the enum constant to a boolean}}
    return a;

  if(1 << 5) // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
    return a;

  // Don't warn in macros.
  return SHIFT(1, a);
}

int GH64356(int arg) {
  if ((arg == 1) && (1 == 1)) return 1;
    return 0;

  if ((64 > 32) && (32 < 64))
    return 2;

  if ((1 == 1) && (arg == 1)) return 1;
    return 0;
}
