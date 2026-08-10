// RUN: %clang_cc1 -fsyntax-only -isystem %S/Inputs/include -verify %s
// expected-no-diagnostics

#include <tgmath.h>

float f;
double d;
long double l;

float complex fc;
double complex dc;
long double complex lc;

// creal

_Static_assert(sizeof(creal(f)) == sizeof(f), "");
_Static_assert(sizeof(creal(d)) == sizeof(d), "");
_Static_assert(sizeof(creal(l)) == sizeof(l), "");

_Static_assert(sizeof(creal(fc)) == sizeof(f), "");
_Static_assert(sizeof(creal(dc)) == sizeof(d), "");
_Static_assert(sizeof(creal(lc)) == sizeof(l), "");

// fabs

_Static_assert(sizeof(fabs(f)) == sizeof(f), "");
_Static_assert(sizeof(fabs(d)) == sizeof(d), "");
_Static_assert(sizeof(fabs(l)) == sizeof(l), "");

_Static_assert(sizeof(fabs(fc)) == sizeof(f), "");
_Static_assert(sizeof(fabs(dc)) == sizeof(d), "");
_Static_assert(sizeof(fabs(lc)) == sizeof(l), "");

// logb

_Static_assert(sizeof(logb(f)) == sizeof(f), "");
_Static_assert(sizeof(logb(d)) == sizeof(d), "");
_Static_assert(sizeof(logb(l)) == sizeof(l), "");
