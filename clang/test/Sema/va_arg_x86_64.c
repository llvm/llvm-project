// RUN: %clang_cc1 -fsyntax-only -verify -triple=x86_64-unknown-freebsd7.0 %s
// expected-no-diagnostics

// PR2631
char* foo(char *fmt, __builtin_va_list ap)
{
  return __builtin_va_arg((ap), char *);
}

// PR2692
typedef __builtin_va_list va_list;
static void f (char * (*g) (char **, int), char **p, ...) {
  char *s;
  va_list v;
  s = g (p, __builtin_va_arg(v, int));
}
