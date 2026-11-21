// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

#if !__has_builtin(__builtin_iseqsig)
#error "__builtin_iseqsig is not available"
#endif

int ok(float f, double d, long double ld, int i) {
  return __builtin_iseqsig(f, f) + __builtin_iseqsig(d, f) +
         __builtin_iseqsig(ld, d) + __builtin_iseqsig(d, i) +
         __builtin_iseqsig(i, ld);
}

int bad(int i, float f) {
  return __builtin_iseqsig(i, i) + // expected-error {{ordered compare requires two args of floating point type ('int' and 'int')}}
         __builtin_iseqsig(f) +    // expected-error {{too few arguments to function call, expected 2, have 1}}
         __builtin_iseqsig(f, f, f); // expected-error {{too many arguments to function call, expected 2, have 3}}
}
