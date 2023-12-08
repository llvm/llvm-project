// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

int i;
int a[] = {0};
struct { int i; } s;

int *array[] = {&i, a, &s.i};

extern void f(void);
void (*f_addr)(void) = &f;
