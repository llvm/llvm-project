// RUN: %clang_analyze_cc1 -std=c++23 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();
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

int assume_and_fallthrough_at_the_same_attrstmt(int a, int b) {
  [[assume(a == 2)]];
  clang_analyzer_dump(a); // expected-warning {{2 S32b}}
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<int a>}} FIXME: We shouldn't have this dump.
  switch (a) {
    case 2:
      [[fallthrough, assume(b == 30)]];
    case 4: {
      clang_analyzer_dump(b); // expected-warning {{30 S32b}}
      // expected-warning-re@-1 {{reg_${{[0-9]+}}<int b>}} FIXME: We shouldn't have this dump.
      return b;
    }
  }

  // This code should be unreachable.
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}

  [[assume(false)]]; // This should definitely make it so.
  clang_analyzer_warnIfReached(); // no-warning

  return 0;
}
