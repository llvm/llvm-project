// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -verify %s

typedef __typeof(sizeof(int)) size_t;

void *operator new(size_t size, void *ptr);
void *operator new[](size_t size, void *ptr);
void *operator new(size_t, void *, int);

void clang_analyzer_eval(bool);

struct B {
  virtual void foo();
};

struct D : public B {
  virtual void foo() override {}
};

void test_ub() {
  // FIXME: Potentially warn because this code is pretty weird.
  B b;
  new (&b) D;
  b.foo(); // no-crash
}

void test_non_ub() {
  char c[sizeof(D)]; // Should be enough storage.
  new (c) D;
  ((B *)c)->foo(); // no-crash
}

struct V {
  virtual int id() { return 1; }
  ~V() {}
};
struct W : V {
  int id() override { return 2; }
  ~W() {}
};

void array_element() {
  V A[2];
  V *P = &A[1];
  new (&A[1]) W;
  clang_analyzer_eval(P->id() == 2); // expected-warning{{TRUE}}
}

struct TwoMembers {
  V First;
  V Second;
};

void restore_through_byte_offset() {
  TwoMembers S;
  new (&S.Second) W;
  new (reinterpret_cast<char *>(&S) + sizeof(V)) V;
  clang_analyzer_eval(S.Second.id() == 1); // expected-warning{{TRUE}}
}

void restore_through_opaque_operator_new() {
  V v;
  new (&v) W;
  new (&v, 1) V;
  clang_analyzer_eval(v.id() == 1); // expected-warning{{TRUE}}
}

void restore_through_array_placement_new() {
  V v;
  new (&v) W;
  new (&v) V[1];
  clang_analyzer_eval(v.id() == 1); // expected-warning{{TRUE}}
}

void after_base_destructor() {
  W w;
  w.~W();
  V *P = &w;
  clang_analyzer_eval(P->id() == 2); // expected-warning{{TRUE}}
  new (&w) W;
}

struct E {
  E() {}
};
struct WithBase : E {
  WithBase(int) {}
  virtual int id() { return 1; }
};
struct Inheriting : WithBase {
  using WithBase::WithBase;
  int id() override { return 2; }
};

void after_inherited_constructor() {
  Inheriting I(0);
  WithBase *P = &I;
  clang_analyzer_eval(P->id() == 2); // expected-warning{{TRUE}}
}
