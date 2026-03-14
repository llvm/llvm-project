// RUN: %clang_analyze_cc1 -analyzer-checker=core,apiModeling.TrustReturnsNonnull -verify %s

int *foo() __attribute__((returns_nonnull));

int *foo_no_attribute();

int test_foo() {
  int *x = foo();
  if (x) {}
  return *x; // no-warning
}

int test_foo_no_attribute() {
  int *x = foo_no_attribute();
  if (x) {}
  return *x;  // expected-warning{{Dereference of null pointer}}
}

void test(void *(*f)(void)) {
  f();  // Shouldn't crash compiler
}
