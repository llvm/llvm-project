// RUN: %clang_cc1 -verify %s

// These must be the last lines in this test.
// expected-error@+1{{unterminated}} expected-error@+1 2{{expected}}
int i = __has_warning(
