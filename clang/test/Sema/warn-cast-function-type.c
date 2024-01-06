// RUN: %clang_cc1 %s -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify
// RUN: %clang_cc1 %s -fsyntax-only -Wextra -Wno-cast-function-type-strict -verify

int x(long);

typedef int (f1)(long);
typedef int (f2)(void*);
typedef int (f3)();
typedef void (f4)();
typedef void (f5)(void);
typedef int (f6)(long, int);
typedef int (f7)(long,...);

f1 *a;
f2 *b;
f3 *c;
f4 *d;
f5 *e;
f6 *f;
f7 *g;

void foo(void) {
  a = (f1 *)x;
  b = (f2 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f2 *' (aka 'int (*)(void *)') converts to incompatible function type}} */
  c = (f3 *)x;
  d = (f4 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f4 *' (aka 'void (*)()') converts to incompatible function type}} */
  e = (f5 *)x;
  f = (f6 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f6 *' (aka 'int (*)(long, int)') converts to incompatible function type}} */
  g = (f7 *)x;
}
