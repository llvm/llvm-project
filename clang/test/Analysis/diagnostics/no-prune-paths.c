// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -analyzer-config prune-paths=false -DNPRUNE=1 -verify %s

// "prune-paths" is a debug option only; this is just a simple test to see that
// it's being honored.

void helper() {
  extern void foo();
  foo();
}

void test() {
  helper();
#if NPRUNE
  // expected-note@-2 {{Calling 'helper'}}
  // expected-note@-3 {{Returning from 'helper'}}
#endif

  *(volatile int *)0 = 1; // expected-warning{{indirection of null pointer will be deleted, not trap}} expected-note{{consider using __builtin_trap()}} expected-warning {{Dereference of null pointer}} expected-note {{Dereference of null pointer}}
}
