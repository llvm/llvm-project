// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wint-in-bool-context %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall %s
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
  r = a << a;    // expected-warning {{converting the result of '<<' to a boolean; did you mean '(a << a) != 0'?}}
  r = MM;        // expected-warning {{converting the result of '<<' to a boolean; did you mean '(a << a) != 0'?}}
  r = (1 << 7);  // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 2UL << 2;  // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 0 << a;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to false}}
  r = 0 << 2;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to false}}
  r = 1 << 0;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 1 << 2;    // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 1ULL << 2; // expected-warning {{converting the result of '<<' to a boolean always evaluates to true}}
  r = 2 << b;    // expected-warning {{converting the result of '<<' to a boolean; did you mean '(2 << b) != 0'?}}
  r = (unsigned)(2 << b);
  r = b << 7;
  r = (1 << a); // expected-warning {{converting the result of '<<' to a boolean; did you mean '(1 << a) != 0'?}}
  r = TWO << a; // expected-warning {{converting the result of '<<' to a boolean; did you mean '(2 << a) != 0'?}}
  r = a << 7;   // expected-warning {{converting the result of '<<' to a boolean; did you mean '(a << 7) != 0'?}}
  r = ONE << a; // expected-warning {{converting the result of '<<' to a boolean; did you mean '(1 << a) != 0'?}}
  if (TWO << a) // expected-warning {{converting the result of '<<' to a boolean; did you mean '(2 << a) != 0'?}}
    return a;

  for (a = 0; 1 << a; a++) // expected-warning {{converting the result of '<<' to a boolean; did you mean '(1 << a) != 0'?}}
    ;

  if (a << TWO) // expected-warning {{converting the result of '<<' to a boolean; did you mean '(a << 2) != 0'?}}
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
