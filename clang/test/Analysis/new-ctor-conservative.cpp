// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-allocator-inlining=true -std=c++11 -verify %s

void clang_analyzer_eval(bool);

struct S {
  int x;
  S() : x(1) {}
  ~S() {}
};

void checkConstructorInlining() {
  S *s = new S;
  clang_analyzer_eval(s->x == 1); // expected-warning{{TRUE}}
}
