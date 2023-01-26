// RUN: %clang_cc1 %s -fsyntax-only -Wcast-function-type -verify
// RUN: %clang_cc1 %s -fsyntax-only -Wcast-function-type-strict -verify


int t(int array[static 12]);
int u(int i);
const int v(int i);
int x(long);

typedef int (f1)(long);
typedef int (f2)(void*);
typedef int (f3)();
typedef void (f4)();
typedef void (f5)(void);
typedef int (f6)(long, int);
typedef int (f7)(long,...);
typedef int (f8)(int *);
typedef int (f9)(const int);
typedef int (f10)(int);

f1 *a;
f2 *b;
f3 *c;
f4 *d;
f5 *e;
f6 *f;
f7 *g;
f8 *h;
f9 *i;
f10 *j;

void foo(void) {
  a = (f1 *)x;
  b = (f2 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f2 *' (aka 'int (*)(void *)') converts to incompatible function type}} */
  c = (f3 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f3 *' (aka 'int (*)()') converts to incompatible function type}} */
  d = (f4 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f4 *' (aka 'void (*)()') converts to incompatible function type}} */
  e = (f5 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f5 *' (aka 'void (*)(void)') converts to incompatible function type}} */
  f = (f6 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f6 *' (aka 'int (*)(long, int)') converts to incompatible function type}} */
  g = (f7 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f7 *' (aka 'int (*)(long, ...)') converts to incompatible function type}} */
  h = (f8 *)t;
  i = (f9 *)u;
  j = (f10 *)v; /* expected-warning {{cast from 'const int (*)(int)' to 'f10 *' (aka 'int (*)(int)') converts to incompatible function type}} */
}
