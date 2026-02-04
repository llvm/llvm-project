// RUN: %clang_cc1 -triple i386-unknown-unknown -Eonly -verify %s

// expected-error@+2 {{builtin feature check macro requires a parenthesized identifier}}
// expected-error@+1 {{expected value in expression}}
#if __has_c_attribute(clang::
#endif

// expected-error@+1 {{builtin feature check macro requires a parenthesized identifier}}
__has_c_attribute(clang::
