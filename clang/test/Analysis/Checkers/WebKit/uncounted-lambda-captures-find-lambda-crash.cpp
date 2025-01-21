// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.UncountedLambdaCapturesChecker -verify %s
// expected-no-diagnostics

struct Foo {
  int x;
  int y;
  Foo(int x, int y) : x(x) , y(y) { }
  ~Foo() { }
};

Foo bar(const Foo&);
void foo() {
  int x = 7;
  int y = 5;
  bar(Foo(x, y));
}
