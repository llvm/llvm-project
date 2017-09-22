// RUN: %clang_cc1 -fsyntax-only -verify %s

void escapefunc(int *);
void noescapefunc(__attribute__((noescape)) int *);
void (*escapefuncptr)(int *);
void (*noescapefuncptr)(__attribute__((noescape)) int *);

void func_ne(__attribute__((noescape)) int *, int *);
void func_en(int *, __attribute__((noescape)) int *);

void (*funcptr_ee)(int *, int *);
void (*funcptr_nn)(__attribute__((noescape)) int *, __attribute__((noescape)) int *);

void test0(int c) {
  escapefuncptr = &escapefunc;
  escapefuncptr = &noescapefunc;
  noescapefuncptr = &escapefunc; // expected-warning {{incompatible function pointer types assigning to 'void (*)(__attribute__((noescape)) int *)' from 'void (*)(int *)'}}
  noescapefuncptr = &noescapefunc;

  escapefuncptr = c ? &escapefunc : &noescapefunc;
  noescapefuncptr = c ? &escapefunc : &noescapefunc; // expected-warning {{incompatible function pointer types assigning to 'void (*)(__attribute__((noescape)) int *)' from 'void (*)(int *)'}}

  funcptr_ee = c ? &func_ne : &func_en;
  funcptr_nn = c ? &func_ne : &func_en; // expected-warning {{incompatible function pointer types assigning to 'void (*)(__attribute__((noescape)) int *, __attribute__((noescape)) int *)' from 'void (*)(int *, int *)'}}
}
