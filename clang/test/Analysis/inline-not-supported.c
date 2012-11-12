// RUN: %clang_cc1 -fblocks -analyze -analyzer-checker=core -verify %s

// For now, don't inline varargs.
void foo(int *x, ...) {
  *x = 1;
}

void bar() {
  foo(0, 2); // no-warning
}

// For now, don't inline vararg blocks.
void (^baz)(int *x, ...) = ^(int *x, ...) { *x = 1; };

void taz() {
  baz(0, 2); // no-warning
}

// For now, don't inline global blocks.
void (^qux)(int *p) = ^(int *p) { *p = 1; };
void test_qux() {
  qux(0); // no-warning
}


void test_analyzer_is_running() {
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}
