// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config widen-loops=true -analyzer-disable-retry-exhausted -verify %s

void clang_analyzer_eval(bool);
void clang_analyzer_dump(int);

// 'this' pointer is not an lvalue, we should not invalidate it.
namespace this_pointer_after_loop_widen {
class A {
public:
  A() {
    int count = 10;
    do {
    } while (count--);
  }
};

void goo(A a);
void test_temporary_object() {
  goo(A()); // no-crash
}

struct B {
  int mem;
  B() : mem(0) {
    for (int i = 0; i < 10; ++i) {
    }
    mem = 0;
  }
};

void test_ctor() {
  B b;
  clang_analyzer_eval(b.mem == 0); // expected-warning{{TRUE}}
}

struct C {
  int mem;
  C() : mem(0) {}
  void set() {
    for (int i = 0; i < 10; ++i) {
    }
    mem = 10;
  }
};

void test_method() {
  C c;
  clang_analyzer_eval(c.mem == 0); // expected-warning{{TRUE}}
  c.set();
  clang_analyzer_eval(c.mem == 10); // expected-warning{{TRUE}}
}

struct D {
  int mem;
  D() : mem(0) {}
  void set() {
    for (int i = 0; i < 10; ++i) {
    }
    mem = 10;
  }
};

void test_new() {
  D *d = new D;
  clang_analyzer_eval(d->mem == 0); // expected-warning{{TRUE}}
  d->set();
  clang_analyzer_eval(d->mem == 10); // expected-warning{{TRUE}}
}

struct E {
  int mem;
  E() : mem(0) {}
  void set() {
    for (int i = 0; i < 10; ++i) {
    }
    setAux();
  }
  void setAux() {
    this->mem = 10;
  }
};

void test_chained_method_call() {
  E e;
  e.set();
  clang_analyzer_eval(e.mem == 10); // expected-warning{{TRUE}}
}
} // namespace this_pointer_after_loop_widen
