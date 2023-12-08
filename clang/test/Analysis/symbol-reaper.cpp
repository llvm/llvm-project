// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_warnOnDeadSymbol(int);

namespace test_dead_region_with_live_subregion_in_environment {
int glob;

struct A {
  int x;

  void foo() {
    // FIXME: Maybe just let clang_analyzer_eval() work within callees already?
    // The glob variable shouldn't keep our symbol alive because
    // 'x != 0' is concrete 'true'.
    glob = (x != 0);
  }
};

void test_A(A a) {
  if (a.x == 0)
    return;

  clang_analyzer_warnOnDeadSymbol(a.x);

  // What we're testing is that a.x is alive until foo() exits.
  a.foo(); // no-warning // (i.e., no 'SYMBOL DEAD' yet)

  // Let's see if constraints on a.x were known within foo().
  clang_analyzer_eval(glob); // expected-warning{{TRUE}}
                             // expected-warning@-1{{SYMBOL DEAD}}
}

struct B {
  A a;
  int y;
};

A &noop(A &a) {
  // This function ensures that the 'b' expression within its argument
  // would be cleaned up before its call, so that only 'b.a' remains
  // in the Environment.
  return a;
}


void test_B(B b) {
  if (b.a.x == 0)
    return;

  clang_analyzer_warnOnDeadSymbol(b.a.x);

  // What we're testing is that b.a.x is alive until foo() exits.
  noop(b.a).foo(); // no-warning // (i.e., no 'SYMBOL DEAD' yet)

  // Let's see if constraints on a.x were known within foo().
  clang_analyzer_eval(glob); // expected-warning{{TRUE}}
                             // expected-warning@-1{{SYMBOL DEAD}}
}
} // namespace test_dead_region_with_live_subregion_in_environment
