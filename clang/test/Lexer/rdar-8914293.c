// RUN: %clang_cc1 -fsyntax-only -verify %s

// rdar://8914293
// We want be compatible with gcc and warn, not error.

/* expected-warning {{missing terminating}} */ #define FOO "foo
/* expected-warning {{missing terminating}} */ #define KOO 'k
