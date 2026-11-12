// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -analyzer-config prune-paths=false -DNPRUNE=1 -verify %s

// "prune-paths" is a debug option only; this is just a simple test to see that
// it's being honored.

void helper(void) {
  extern void foo(void);
  foo();
}

void test(void) {
  helper();
#if NPRUNE
  // expected-note@-2 {{Calling 'helper'}}
  // expected-note@-3 {{Returning from 'helper'}}
#endif

  *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer}}
}
