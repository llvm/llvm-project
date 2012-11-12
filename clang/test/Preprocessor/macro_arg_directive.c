// RUN: %clang_cc1 %s -fsyntax-only -verify

#define a(x) enum { x }
a(n =
#undef a
#define a 5
  a);
_Static_assert(n == 5, "");

// header1.h
void fail(const char *);
#define MUNCH(...) \
 ({ int result = 0; __VA_ARGS__; if (!result) { fail(#__VA_ARGS__); }; result })

static inline int f(int k) {
  return MUNCH( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{returning 'void'}}
    if (k < 3)
      result = 24;
    else if (k > 4)
      result = k - 4;
}

#include "macro_arg_directive.h" // expected-error {{embedding a #include directive within macro arguments is not supported}}

int g(int k) {
  return f(k) + f(k-1));
}
