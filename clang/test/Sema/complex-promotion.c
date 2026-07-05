// RUN: %clang_cc1 %s -verify -fsyntax-only
// expected-no-diagnostics

float a;

int b[__builtin_classify_type(a + 1i) == 9 ? 1 : -1];
int c[__builtin_classify_type(1i + a) == 9 ? 1 : -1];

double d;
__typeof__ (d + 1i) e;

int f[sizeof(e) == 2 * sizeof(double) ? 1 : -1];

int g;
int h[__builtin_classify_type(g + 1.0i) == 9 ? 1 : -1];
int i[__builtin_classify_type(1.0i + a) == 9 ? 1 : -1];
