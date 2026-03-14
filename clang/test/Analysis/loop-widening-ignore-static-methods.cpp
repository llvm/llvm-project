// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config widen-loops=true -analyzer-max-loop 2 %s
// REQUIRES: asserts
// expected-no-diagnostics
//
// This test checks that the loop-widening code ignores static methods.  If that is not the
// case, then an assertion will trigger.

class Test {
  static void foo() {
    for (;;) {}
  }
};
