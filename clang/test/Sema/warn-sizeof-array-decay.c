// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(int x) {
  char foo[10];
  int bar[20];
  char qux[30];

  (void)sizeof(bar + 10); // expected-warning{{sizeof on pointer operation will return size of 'int *' instead of 'int [20]'}}
  (void)sizeof(foo - 20); // expected-warning{{sizeof on pointer operation will return size of 'char *' instead of 'char [10]'}}
  (void)sizeof(bar - x); // expected-warning{{sizeof on pointer operation will return size of 'int *' instead of 'int [20]'}}
  (void)sizeof(foo + x); // expected-warning{{sizeof on pointer operation will return size of 'char *' instead of 'char [10]'}}

  // This is ptrdiff_t.
  (void)sizeof(foo - qux); // no-warning

  (void)sizeof(foo, x); // no-warning
  (void)sizeof(x, foo); // expected-warning{{sizeof on pointer operation will return size of 'char *' instead of 'char [10]'}}
}
