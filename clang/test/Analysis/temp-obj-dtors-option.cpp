// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-temp-dtor-inlining=false -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-temp-dtor-inlining=true -DINLINE -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

struct S {
  int &x;

  S(int &x) : x(x) { ++x; }
  ~S() { --x; }
};

void foo() {
  int x = 0;
  S(x).x += 1;
  clang_analyzer_eval(x == 1);
#ifdef INLINE
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
}
