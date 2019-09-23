// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wint-in-bool-context %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wint-in-bool-context %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wall %s

#define ONE 1
#define TWO 2

#define SHIFT(l, r) l << r

#ifdef __cplusplus
typedef bool boolean;
#else
typedef _Bool boolean;
#endif

int test(int a) {
  boolean r;
  r = (1 << 3); // expected-warning {{converting the result of '<<' to a boolean; did you mean '(1 << 3) != 0'?}}
  r = TWO << 7; // expected-warning {{converting the result of '<<' to a boolean; did you mean '(2 << 7) != 0'?}}
  r = a << 7;   // expected-warning {{converting the result of '<<' to a boolean; did you mean '(a << 7) != 0'?}}
  r = ONE << a; // expected-warning {{converting the result of '<<' to a boolean; did you mean '(1 << a) != 0'?}}
  if (TWO << 4) // expected-warning {{converting the result of '<<' to a boolean; did you mean '(2 << 4) != 0'?}}
    return a;

  if (a << TWO) // expected-warning {{converting the result of '<<' to a boolean; did you mean '(a << 2) != 0'?}}
    return a;

  // Don't warn in macros.
  return SHIFT(1, a);
}
