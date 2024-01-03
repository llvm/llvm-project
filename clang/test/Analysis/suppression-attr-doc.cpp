// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix \
// RUN:                    -analyzer-disable-checker=core.uninitialized \
// RUN:                    -verify %s

// NOTE: These tests correspond to examples provided in documentation
// of [[clang::suppress]]. If you break them intentionally, it's likely that
// you need to update the documentation!

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

int foo_initial() {
  int *x = nullptr;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

int foo1() {
  int *x = nullptr;
  [[clang::suppress]]
  return *x;  // null pointer dereference warning suppressed here
}

int foo2() {
  [[clang::suppress]] {
    int *x = nullptr;
    return *x;  // null pointer dereference warning suppressed here
  }
}

int bar_initial(bool coin_flip) {
  int *result = (int *)malloc(sizeof(int));
  if (coin_flip)
    return 1; // There's no warning here YET, but it will show up if the other one is suppressed.

  return *result;  // expected-warning{{Potential leak of memory pointed to by 'result'}}
}

int bar1(bool coin_flip) {
  __attribute__((suppress))
  int *result = (int *)malloc(sizeof(int));
  if (coin_flip)
    return 1;  // warning about this leak path is suppressed

  return *result;  // warning about this leak path also suppressed
}

int bar2(bool coin_flip) {
  int *result = (int *)malloc(sizeof(int));
  if (coin_flip)
    return 1;  // expected-warning{{Potential leak of memory pointed to by 'result'}}

  __attribute__((suppress))
  return *result;  // leak warning is suppressed only on this path
}
