// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config eagerly-assume=false -verify %s
// UNSUPPORTED: z3

template <typename T>
void clang_analyzer_value(T x);
void clang_analyzer_value();
template <typename T1, typename T2>
void clang_analyzer_value(T1 x, T2 y);

void test1(char x) {
  clang_analyzer_value(x); // expected-warning{{8s:{ [-128, 127] }}}
  if (x > 42)
    clang_analyzer_value(x); // expected-warning{{8s:{ [43, 127] }}}
  if (x == 42)
    clang_analyzer_value(x); // expected-warning{{8s:42}}
}

void test2(short x) {
  clang_analyzer_value(x); // expected-warning{{16s:{ [-32768, 32767] }}}
  if (x < 4200)
    clang_analyzer_value(x); // expected-warning{{16s:{ [-32768, 4199] }}}
  if (x == 4200)
    clang_analyzer_value(x); // expected-warning{{16s:4200}}
}

void test3(unsigned long long x) {
  clang_analyzer_value(x); // expected-warning{{64u:{ [0, 18446744073709551615] }}}
  if (x != 42000000)
    clang_analyzer_value(x); // expected-warning{{64u:{ [0, 41999999], [42000001, 18446744073709551615] }}}
  if (x == 18446744073709551615ull)
    clang_analyzer_value(x); // expected-warning{{64u:18446744073709551615}}
}

struct S {};
void test4(S s) {
  clang_analyzer_value(s); // expected-warning{{n/a}}
}

void test5() {
  clang_analyzer_value(); // expected-warning{{Missing argument}}
}

void test6(int x, int y) {
  if (x == 42 && y == 24)
    // Ignore 'y'.
    clang_analyzer_value(x, y); // expected-warning{{32s:42}}
}
