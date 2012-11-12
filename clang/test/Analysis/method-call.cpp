// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=inlining -analyzer-config c++-inlining=constructors -verify %s

void clang_analyzer_eval(bool);


struct A {
  int x;
  A(int a) { x = a; }
  int getx() const { return x; }
};

struct B{
  int x;
};

void testNullObject(A *a) {
  clang_analyzer_eval(a); // expected-warning{{UNKNOWN}}
  (void)a->getx(); // assume we know what we're doing
  clang_analyzer_eval(a); // expected-warning{{TRUE}}
}

void f1() {
  A x(3);
  clang_analyzer_eval(x.getx() == 3); // expected-warning{{TRUE}}
}

void f2() {
  const A &x = A(3);
  clang_analyzer_eval(x.getx() == 3); // expected-warning{{TRUE}}
}

void f3() {
  const A &x = (A)3;
  clang_analyzer_eval(x.getx() == 3); // expected-warning{{TRUE}}
}

void f4() {
  A x = 3;
  clang_analyzer_eval(x.getx() == 3); // expected-warning{{TRUE}}
}

void checkThatCopyConstructorDoesNotInvalidateObjectBeingCopied() {
  B t;
  t.x = 0;
  B t2(t);
  clang_analyzer_eval(t.x == 0); // expected-warning{{TRUE}}
}
