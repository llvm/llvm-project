// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:                    -verify %s

#define nil ((id)0)

void clang_analyzer_eval(int);

struct S {
  int x;
  S();
};

@interface I
@property S s;
@end

void foo() {
  // This produces a zero-initialized structure.
  // FIXME: This very fact does deserve the warning, because zero-initialized
  // structures aren't always valid in C++. It's particularly bad when the
  // object has a vtable.
  S s = ((I *)nil).s;
  clang_analyzer_eval(s.x == 0); // expected-warning{{TRUE}}
}
