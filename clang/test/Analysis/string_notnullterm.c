// RUN: %clang_analyze_cc1 -verify %s -Wno-null-dereference \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring.NotNullTerminated \
// RUN:   -analyzer-checker=debug.ExprInspection

char *strcpy(char *restrict s1, const char *restrict s2);

void strcpy_fn(char *x) {
  strcpy(x, (char*)&strcpy_fn); // expected-warning{{Argument to string copy function is the address of the function 'strcpy_fn', which is not a null-terminated string}}
}
