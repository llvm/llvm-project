// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s

int printf(char const *, ...);

void test(void) {
  // size_t
  printf("%zu", (double)42); // expected-warning {{format specifies type 'size_t' (aka 'unsigned long') but the argument has type 'double'}}

  // intmax_t / uintmax_t
  printf("%jd", (double)42); // expected-warning {{format specifies type 'intmax_t' (aka 'long') but the argument has type 'double'}}
  printf("%ju", (double)42); // expected-warning {{format specifies type 'uintmax_t' (aka 'unsigned long') but the argument has type 'double'}}

  // ptrdiff_t
  printf("%td", (double)42); // expected-warning {{format specifies type 'ptrdiff_t' (aka 'long') but the argument has type 'double'}}
}

void test_writeback(void) {
  printf("%jn", (long*)0); // no-warning
  printf("%jn", (unsigned long*)0); // no-warning
  printf("%jn", (int*)0); // expected-warning{{format specifies type 'intmax_t *' (aka 'long *') but the argument has type 'int *'}}

  printf("%zn", (long*)0); // no-warning
  // FIXME: Warn about %zn with non-ssize_t argument.

  printf("%tn", (long*)0); // no-warning
  printf("%tn", (unsigned long*)0); // no-warning
  printf("%tn", (int*)0); // expected-warning{{format specifies type 'ptrdiff_t *' (aka 'long *') but the argument has type 'int *'}}
}
