// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=debug.ExprInspection -verify %s

// Self-tests for the debug.ExprInspection checker.

void clang_analyzer_denote(int x, const char *str);
void clang_analyzer_express(int x);

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
