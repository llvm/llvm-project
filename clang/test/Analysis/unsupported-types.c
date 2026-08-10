// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -triple x86_64-unknown-linux -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -triple powerpc64-linux-gnu -verify %s

#define _Complex_I      (__extension__ 1.0iF)

void clang_analyzer_eval(int);

void complex_float(double _Complex x, double _Complex y) {
  clang_analyzer_eval(x == y); // expected-warning{{UNKNOWN}}
  if (x != 1.0 + 3.0 * _Complex_I && y != 1.0 - 4.0 * _Complex_I)
    return
  clang_analyzer_eval(x == y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x + y == 2.0 - 1.0 * _Complex_I); // expected-warning{{UNKNOWN}}
}

void complex_int(int _Complex x, int _Complex y) {
  clang_analyzer_eval(x == y); // expected-warning{{UNKNOWN}}
  if (x != 1.0 + 3.0 * _Complex_I && y != 1.0 - 4.0 * _Complex_I)
    return
  clang_analyzer_eval(x == y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x + y == 2.0 - 1.0 * _Complex_I); // expected-warning{{UNKNOWN}}
}

void longdouble_float(long double x, long double y) {
  clang_analyzer_eval(x == y); // expected-warning{{UNKNOWN}}
  if (x != 0.0L && y != 1.0L)
    return
  clang_analyzer_eval(x == y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x + y == 1.0L); // expected-warning{{UNKNOWN}}
}
