// RUN: %clang_cc1 -verify %s

// These must be the last lines in this test.
// expected-error@+1{{too few arguments}}
int i = __has_warning();
