// RUN: %clang_cc1 -fsyntax-only -Wc++98-compat-pedantic -std=c++11 -verify %s

// The following line isn't terminated, don't fix it.
void foo() {} // expected-warning{{C++98 requires newline at end of file}}