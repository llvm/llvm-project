// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c89
// expected-no-diagnostics
// rdar://6095180

struct s { char c[17]; };
extern struct s foo(void);

struct s a, b, c;

int A[sizeof((foo().c)) == 17 ? 1 : -1];
int B[sizeof((a.c)) == 17 ? 1 : -1];


// comma does not promote array/function in c90 unless they are lvalues.
int W[sizeof(0, a.c) == sizeof(char*) ? 1 : -1];
int X[sizeof(0, (foo().c)) == 17 ? 1 : -1];
int Y[sizeof(0, (a,b).c) == 17 ? 1 : -1];
int Z[sizeof(0, (a=b).c) == 17 ? 1 : -1];
