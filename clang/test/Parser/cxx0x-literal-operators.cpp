// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

void operator "" (const char *); // expected-error {{expected identifier}}
void operator "k" foo(const char *); // \
  expected-error {{string literal after 'operator' must be '""'}} \
  expected-warning{{user-defined literal suffixes not starting with '_' are reserved}}
void operator "" tester (const char *); // \
  expected-warning{{user-defined literal suffixes not starting with '_' are reserved}}
