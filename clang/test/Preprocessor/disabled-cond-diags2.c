// RUN: %clang_cc1 -Eonly -verify %s

#if 0
#if 1
#endif junk // shouldn't produce diagnostics
#endif

#if 0
#endif junk // expected-warning{{extra tokens at end of #endif directive}}

#if 1 junk  // expected-error{{token is not a valid binary operator in a preprocessor subexpression}}
#X          // shouldn't produce diagnostics (block #if condition not valid, so skipped)
#else
#X          // expected-error{{invalid preprocessing directive}}
#endif

#if 0
// diagnostics should not be produced until final #endif
#X
#include
#if 1 junk
#else junk
#endif junk
#line -2
#error
#warning
#endif junk // expected-warning{{extra tokens at end of #endif directive}}
