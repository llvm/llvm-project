// RUN: %clang_cc1 -triple x86_64-linux -verify -Wno-unknown-pragmas %s
// RUN: %clang_cc1 -triple x86_64-linux -verify -fexperimental-new-constant-interpreter -Wno-unknown-pragmas %s
// expected-no-diagnostics

#pragma STDC FENV_ROUND FE_TONEARESTFROMZERO
static_assert(__builtin_fdim(__DBL_EPSILON__ / 2., -1.) == 1. + __DBL_EPSILON__, "");
static_assert(__builtin_fma(0.5, __DBL_EPSILON__, 1.0) == 1. + __DBL_EPSILON__, "");
static_assert(__builtin_scalbn(1.0 + __DBL_EPSILON__ / 2., 0) == 1. + __DBL_EPSILON__, "");

#pragma STDC FENV_ROUND FE_TONEAREST
static_assert(__builtin_fdim(__DBL_EPSILON__ / 2., -1.) == 1., "");
static_assert(__builtin_fma(0.5, __DBL_EPSILON__, 1.0) == 1., "");
static_assert(__builtin_scalbn(1.0 + __DBL_EPSILON__ / 2., 0) == 1., "");
