// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-store=region -verify %s

void clang_analyzer_eval(bool);

bool PR14634(int x) {
  double y = (double)x;
  return !y;
}

bool PR14634_implicit(int x) {
  double y = (double)x;
  return y;
}

void intAsBoolAsSwitchCondition(int c) {
  switch ((bool)c) { // expected-warning {{switch condition has boolean value}}
  case 0:
    break;
  }

  switch ((int)(bool)c) { // no-warning
    case 0:
      break;
  }
}

int *&castToIntPtrLValueRef(char *p) {
  return (int *&)*(int *)p;
}
bool testCastToIntPtrLValueRef(char *p, int *s) {
  return castToIntPtrLValueRef(p) != s; // no-crash
}

int *&&castToIntPtrRValueRef(char *p) {
  return (int *&&)*(int *)p;
}
bool testCastToIntPtrRValueRef(char *p, int *s) {
  return castToIntPtrRValueRef(p) != s; // no-crash
}

bool retrievePointerFromBoolean(int *p) {
  bool q;
  *reinterpret_cast<int **>(&q) = p;
  return q;
}

namespace base_to_derived {
struct A {};
struct B : public A{};

void foo(A* a) {
  B* b = (B* ) a;
  A* a2 = (A *) b;
  clang_analyzer_eval(a2 == a); // expected-warning{{TRUE}}
}
}

namespace base_to_derived_double_inheritance {
struct A {
  int x;
};
struct B {
  int y;
};
struct C : A, B {};

void foo(B *b) {
  C *c = (C *)b;
  b->y = 1;
  clang_analyzer_eval(c->x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c->y); // expected-warning{{TRUE}}
}
}

