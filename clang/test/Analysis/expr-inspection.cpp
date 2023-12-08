// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config support-symbolic-integer-casts=false \
// RUN:    -verify %s

// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config support-symbolic-integer-casts=true \
// RUN:    -verify %s

// Self-tests for the debug.ExprInspection checker.

template <typename T>
void clang_analyzer_denote(T x, const char *str);
template <typename T>
void clang_analyzer_express(T x);

// Invalid declarations to test basic correctness checks.
void clang_analyzer_denote();
void clang_analyzer_denote(int x);
void clang_analyzer_express();

void foo(int x, unsigned y) {
  clang_analyzer_denote(); // expected-warning{{clang_analyzer_denote() requires a symbol and a string literal}}
  clang_analyzer_express(); // expected-warning{{Missing argument}}

  clang_analyzer_denote(x); // expected-warning{{clang_analyzer_denote() requires a symbol and a string literal}}
  clang_analyzer_express(x); // expected-warning{{Unable to express}}

  clang_analyzer_denote(x, "$x");
  clang_analyzer_express(-x); // expected-warning{{-$x}}

  clang_analyzer_denote(y, "$y");
  clang_analyzer_express(x + y); // expected-warning{{$x + $y}}

  clang_analyzer_denote(1, "$z");     // expected-warning{{Not a symbol}}
  clang_analyzer_express(1);     // expected-warning{{Not a symbol}}

  clang_analyzer_denote(x + 1, "$w");
  clang_analyzer_express(x + 1); // expected-warning{{$w}}
  clang_analyzer_express(y + 1); // expected-warning{{$y + 1U}}
}
