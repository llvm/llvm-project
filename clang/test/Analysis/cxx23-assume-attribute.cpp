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
  // FIXME notes
  // Currently, if this test is run without the core.builtin.Builtin checker, the above function with the __builtin_assume behaves identically to the following test
  // i.e. calls to `clang_analyzer_dump` result in "extraneous"  prints of the SVal(s) `reg<int b> ...`
  // as opposed to 4 or 10
  // which likely implies the Program State(s) did not get narrowed.
  // A new checker is likely needed to be implemented to properly handle the expressions within `[[assume]]` to eliminate the states where `b` is not narrowed.

  [[assume(a > 10 ? b == 4 : b == 10)]];
  clang_analyzer_value(a);
  // expected-warning@-1 {{[-2147483648, 10]}}
  // expected-warning@-2 {{[11, 2147483647]}}

  clang_analyzer_dump(b); // expected-warning {{4}} expected-warning {{10}}
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<int b>}} FIXME: We shouldn't have this dump.

  if (a > 20) {
    clang_analyzer_dump(b + 100); // expected-warning {{104}}
    // expected-warning-re@-1 {{(reg_${{[0-9]+}}<int b>) + 100}} FIXME: We shouldn't have this dump.
    return 2;
  }
  if (a > 10) {
    clang_analyzer_dump(b + 200); // expected-warning {{204}}
    // expected-warning-re@-1 {{(reg_${{[0-9]+}}<int b>) + 200}} FIXME: We shouldn't have this dump.
    return 1;
  }
  clang_analyzer_dump(b + 300); // expected-warning {{310}}
  // expected-warning-re@-1 {{(reg_${{[0-9]+}}<int b>) + 300}} FIXME: We shouldn't have this dump.
  return 0;
}
