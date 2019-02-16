// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-allocator-inlining=true -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;

void *conjure();
void exit(int);

void *operator new(size_t size) throw() {
  void *x = conjure();
  if (x == 0)
    exit(1);
  return x;
}

struct S {
  int x;
  S() : x(1) {}
  ~S() {}
};

void checkNewAndConstructorInlining() {
  S *s = new S;
  // Check that the symbol for 's' is not dying.
  clang_analyzer_eval(s != 0); // expected-warning{{TRUE}}
  // Check that bindings are correct (and also not dying).
  clang_analyzer_eval(s->x == 1); // expected-warning{{TRUE}}
}

struct Sp {
  Sp *p;
  Sp(Sp *p): p(p) {}
  ~Sp() {}
};

void checkNestedNew() {
  Sp *p = new Sp(new Sp(0));
  clang_analyzer_eval(p->p->p == 0); // expected-warning{{TRUE}}
}

void checkNewPOD() {
  int *i = new int;
  clang_analyzer_eval(*i == 0); // expected-warning{{UNKNOWN}}
  int *j = new int();
  clang_analyzer_eval(*j == 0); // expected-warning{{TRUE}}
  int *k = new int(5);
  clang_analyzer_eval(*k == 5); // expected-warning{{TRUE}}
}

void checkTrivialCopy() {
  S s;
  S *t = new S(s); // no-crash
  clang_analyzer_eval(t->x == 1); // expected-warning{{TRUE}}
}
