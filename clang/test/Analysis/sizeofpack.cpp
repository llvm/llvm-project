// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:                    -std=c++14 -verify %s

typedef __typeof(sizeof(int)) size_t;

void clang_analyzer_eval(bool);

template <int... N> size_t foo() {
  return sizeof...(N);
}

void bar() {
  clang_analyzer_eval(foo<>() == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(foo<1, 2, 3>() == 3); // expected-warning{{TRUE}}
}
