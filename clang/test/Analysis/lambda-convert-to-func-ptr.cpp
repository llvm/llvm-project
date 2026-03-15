// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s

void clang_analyzer_eval(bool);

void basic() {
  int (*ret_zero)() = []() { return 0; };
  clang_analyzer_eval(ret_zero() == 0); // expected-warning{{TRUE}}
}

void withParam() {
  int (*add_ten)(int) = [](int b) { return b + 10; };
  clang_analyzer_eval(add_ten(1) == 11); // expected-warning{{TRUE}}
}

int callBack(int (*fp)(int), int x) {
  return fp(x);
}

void passWithFunc() {
  clang_analyzer_eval(callBack([](int x) { return x; }, 5) == 5); // expected-warning{{TRUE}}
}
