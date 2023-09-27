// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify -triple x86_64-apple-darwin %s

// This file tests -Wconstant-conversion, a subcategory of -Wconversion
// which is on by default.

constexpr int nines() { return 99999; }

void too_big_for_char(int param) {
  char warn1 = false ? 0 : 99999;
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}
  char warn2 = false ? 0 : nines();
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}

  char warn3 = param > 0 ? 0 : 99999;
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}
  char warn4 = param > 0 ? 0 : nines();
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}

  char ok1 = true ? 0 : 99999;
  char ok2 = true ? 0 : nines();

  char ok3 = true ? 0 : 99999 + 1;
  char ok4 = true ? 0 : nines() + 1;
}

void test_bitfield() {
  struct S {
    int one_bit : 1;
  } s;

  s.one_bit = 1;    // expected-warning {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
  s.one_bit = true; // no-warning
}

namespace Initializers {
constexpr char ok = true ? 0 : 200;
constexpr char a = 200; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}
char b = 200; // expected-warning {{implicit conversion from 'int' to 'char' changes value from 200 to -56}}
const char c = 200; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}

void f() {
  constexpr char a = 200; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}
  char b = 200; // expected-warning {{implicit conversion from 'int' to 'char' changes value from 200 to -56}}
  const char c = 200; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}
  static char d = 2 * 100; // expected-warning {{implicit conversion from 'int' to 'char' changes value from 200 to -56}}
}

constexpr void g() {
  constexpr char a = 2 * 100; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}
  char b = 2 * 100; // expected-warning {{implicit conversion from 'int' to 'char' changes value from 200 to -56}}
  const char c = 2 * 100; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}
}

consteval void h() {
  char ok = true ? 0 : 200;
  constexpr char a = 200; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}
  char b = 200; // expected-warning {{implicit conversion from 'int' to 'char' changes value from 200 to -56}}
  const char c = 200; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}}
}

template <int N>
int templ() {
  constexpr char a = false ? 129 : N; // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 200 to -56}} \
                                      // expected-warning {{implicit conversion from 'int' to 'const char' changes value from 345 to 89}}
  return 3;
}

void call_templ() {
  int ok = templ<127>();
  int l = templ<3>();
  int m = templ<200>(); // expected-note {{in instantiation of}}
  int n = templ<345>(); // expected-note {{in instantiation of}}
}

template <int a, int b>
constexpr signed char diff = a > b ? a - b : b - a; // expected-warning{{changes value from 201 to -55}} \
                                                    // expected-warning{{changes value from 199 to -57}} \
                                                    // expected-warning{{changes value from 299 to 43}} \
                                                    // expected-warning{{changes value from 301 to 45}}

void test_diff() {
  char ok1 = diff<201, 100>;
  char ok2 = diff<101, 200>;
  char s1 = diff<301, 100>; // expected-note {{in instantiation of}}
  char s2 = diff<101, 300>; // expected-note {{in instantiation of}}
  char w1 = diff<101, 400>; // expected-note {{in instantiation of}}
  char w2 = diff<401, 100>; // expected-note {{in instantiation of}}
}
}
