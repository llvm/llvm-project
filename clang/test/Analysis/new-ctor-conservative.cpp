// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-allocator-inlining=true -std=c++11 -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

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

  // FIXME: Handle big array construction
  clang_analyzer_eval(s[0].x == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(s[1].x == 1); // expected-warning{{UNKNOWN}}

  s = new S[4];
  clang_analyzer_eval(s[0].x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(s[1].x == 1); // expected-warning{{TRUE}}
}

struct NullS {
  NullS() {
    if (this) {}
  }
  NullS(int x) {
    if (!this) {
      clang_analyzer_warnIfReached(); // no-warning
    }
  }
};

void checkNullThis() {
  NullS *nulls = new NullS(); // no-crash
  NullS *nulls2 = new NullS(0);
}
