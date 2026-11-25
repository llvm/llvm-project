// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify -ffp-exception-behavior=strict -fexperimental-strict-floating-point %s
// RUN: %clang_cc1 -fsyntax-only -verify -ffp-exception-behavior=strict -fexperimental-strict-floating-point -fexperimental-new-constant-interpreter %s

#if !__has_builtin(__builtin_iseqsig)
#error "__builtin_iseqsig is not available"
#endif

int ok(float f, double d, long double ld, int i) {
  return __builtin_iseqsig(f, f) + __builtin_iseqsig(d, f) +
         __builtin_iseqsig(ld, d) + __builtin_iseqsig(d, i) +
         __builtin_iseqsig(i, ld);
}

int bad(int i, float f) {
  return __builtin_iseqsig(i, i) + // expected-error {{floating point comparison requires two arguments of floating point type ('int' and 'int')}}
         __builtin_iseqsig(f) +    // expected-error {{too few arguments to function call, expected 2, have 1}}
         __builtin_iseqsig(f, f, f); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

// The values from GCC's builtin-iseqsig tests. A NaN is never equal to
// anything, and +0 and -0 are equal. These are constant contexts, so they
// fold even when FP exceptions are modeled.
#define CASES(ZERO, ONE, INF, NAN, SNAN)                                       \
  __builtin_iseqsig(ZERO, ZERO) && __builtin_iseqsig(ZERO, -ZERO) &&           \
  !__builtin_iseqsig(ZERO, ONE) && !__builtin_iseqsig(-ZERO, ONE) &&           \
  !__builtin_iseqsig(ZERO, INF) && !__builtin_iseqsig(-ZERO, INF) &&           \
  !__builtin_iseqsig(ZERO, NAN) && !__builtin_iseqsig(-ZERO, NAN) &&           \
  __builtin_iseqsig(ONE, ONE) && !__builtin_iseqsig(ONE, ZERO) &&              \
  !__builtin_iseqsig(ONE, -ZERO) && !__builtin_iseqsig(ONE, INF) &&            \
  !__builtin_iseqsig(ONE, NAN) && __builtin_iseqsig(INF, INF) &&               \
  !__builtin_iseqsig(INF, NAN) && !__builtin_iseqsig(NAN, NAN) &&              \
  !__builtin_iseqsig(SNAN, ONE) && !__builtin_iseqsig(ONE, SNAN) &&            \
  !__builtin_iseqsig(SNAN, SNAN)

__extension__ _Static_assert(CASES(0.0f, 1.0f, __builtin_inff(),
                                   __builtin_nanf(""), __builtin_nansf("")),
                             "");
__extension__ _Static_assert(CASES(0.0, 1.0, __builtin_inf(),
                                   __builtin_nan(""), __builtin_nans("")),
                             "");
__extension__ _Static_assert(CASES(0.0L, 1.0L, __builtin_infl(),
                                   __builtin_nanl(""), __builtin_nansl("")),
                             "");
