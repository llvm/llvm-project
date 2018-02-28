// RUN: %clang_analyze_cc1 -analyzer-checker=core -std=c++11 -verify %s

// expected-no-diagnostics

typedef __typeof(sizeof(int)) size_t;

void *operator new(size_t size, void *ptr);

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
