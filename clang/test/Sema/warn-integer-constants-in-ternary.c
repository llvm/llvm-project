// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wtautological-constant-compare %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wtautological-constant-compare %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify %s

#define ONE 1
#define TWO 2

#define TERN(c, l, r) c ? l : r

#ifdef __cplusplus
typedef bool boolean;
#else
typedef _Bool boolean;
#endif

void test(boolean a) {
  boolean r;
  r = a ? (1) : TWO;
  r = a ? 3 : TWO; // expected-warning {{converting the result of '?:' with integer constants to a boolean always evaluates to 'true'}}
  r = a ? -2 : 0;
  r = a ? 3 : -2;  // expected-warning {{converting the result of '?:' with integer constants to a boolean always evaluates to 'true'}}
  r = a ? 0 : TWO;
  r = a ? 3 : ONE; // expected-warning {{converting the result of '?:' with integer constants to a boolean always evaluates to 'true'}}
  r = a ? ONE : 0;
  r = a ? 0 : -0;
  r = a ? 1 : 0;
  r = a ? ONE : 0;
  r = a ? ONE : ONE;
  r = TERN(a, 4, 8);   // expected-warning {{converting the result of '?:' with integer constants to a boolean always evaluates to 'true'}}
  r = TERN(a, -1, -8); // expected-warning {{converting the result of '?:' with integer constants to a boolean always evaluates to 'true'}}
}
