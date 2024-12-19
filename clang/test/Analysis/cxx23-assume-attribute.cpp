// RUN: %clang_analyze_cc1 -std=c++23 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection -verify %s

template <typename T> void clang_analyzer_dump(T);
template <typename T> void clang_analyzer_value(T);

int ternary_in_builtin_assume(int a, int b) {
  __builtin_assume(a > 10 ? b == 4 : b == 10);

  clang_analyzer_value(a);
  // expected-warning@-1 {{[-2147483648, 10]}}
  // expected-warning@-2 {{[11, 2147483647]}}

  clang_analyzer_dump(b); // expected-warning{{4}} expected-warning{{10}}

  if (a > 20) {
    clang_analyzer_dump(b + 100); // expected-warning {{104}}
    return 2;
  }
  if (a > 10) {
    clang_analyzer_dump(b + 200); // expected-warning {{204}}
    return 1;
  }
  clang_analyzer_dump(b + 300); // expected-warning {{310}}
  return 0;
}

// From: https://github.com/llvm/llvm-project/pull/116462#issuecomment-2517853226
int ternary_in_assume(int a, int b) {
  [[assume(a > 10 ? b == 4 : b == 10)]];
  clang_analyzer_value(a);
  // expected-warning@-1 {{[-2147483648, 10]}}
  // expected-warning@-2 {{[11, 2147483647]}}

  clang_analyzer_dump(b); // expected-warning {{4}} expected-warning {{10}}

  if (a > 20) {
    clang_analyzer_dump(b + 100); // expected-warning {{104}}
    return 2;
  }
  if (a > 10) {
    clang_analyzer_dump(b + 200); // expected-warning {{204}}
    return 1;
  }
  clang_analyzer_dump(b + 300); // expected-warning {{310}}
  return 0;
}
