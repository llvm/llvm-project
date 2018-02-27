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

void checkNewPOD() {
  int *i = new int;
  clang_analyzer_eval(*i == 0); // expected-warning{{UNKNOWN}}
  int *j = new int();
  clang_analyzer_eval(*j == 0); // expected-warning{{TRUE}}
  int *k = new int(5);
  clang_analyzer_eval(*k == 5); // expected-warning{{TRUE}}
}

void checkNewArray() {
  S *s = new S[10];
  // FIXME: Should be true once we inline array constructors.
  clang_analyzer_eval(s[0].x == 1); // expected-warning{{UNKNOWN}}
}
