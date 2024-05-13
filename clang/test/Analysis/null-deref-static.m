// RUN: %clang_cc1 -w -fblocks -analyze -analyzer-checker=core,deadcode,alpha.core,debug.ExprInspection -verify %s

void *malloc(unsigned long);
void clang_analyzer_warnIfReached(void);

void test_static_from_block(void) {
  static int *x;
  ^{
    *x; // no-warning
  };
}

void test_static_within_block(void) {
  ^{
    static int *x;
    *x; // expected-warning{{Dereference of null pointer}}
  };
}

void test_static_control_flow(int y) {
  static int *x;
  if (x) {
    // FIXME: Should be reachable.
    clang_analyzer_warnIfReached(); // no-warning
  }
  if (y) {
    // We are not sure if this branch is possible, because the developer
    // may argue that function is always called with y == 1 for the first time.
    // In this case, we can only advise the developer to add assertions
    // for suppressing such path.
    *x; // expected-warning{{Dereference of null pointer}}
  } else {
    x = malloc(1);
  }
}
