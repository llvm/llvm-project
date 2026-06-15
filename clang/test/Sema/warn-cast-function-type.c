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

enum E : long;
int efunc(enum E);

// Produce the underlying `long` type implicitly.
enum E2 { big = __LONG_MAX__ };
int e2func(enum E2);

void foo(void) {
  a = (f1 *)x;
  a = (f1 *)efunc; // enum is just type system sugar, still passed as a long.
  a = (f1 *)e2func; // enum is just type system sugar, still passed as a long.
  b = (f2 *)x; // ABI-compatible: long and void* same size on this target.
  c = (f3 *)x;
  d = (f4 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f4 *' (aka 'void (*)()') converts to incompatible function type}} */
  e = (f5 *)x;
  f = (f6 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f6 *' (aka 'int (*)(long, int)') converts to incompatible function type}} */
  g = (f7 *)x;
}

// Pointer-vs-integral return types: same size so ABI-compatible on default target.
typedef void *(*ptr_ret_fn)(void);
typedef unsigned long (*ul_ret_fn)(void);
ptr_ret_fn pr;
ul_ret_fn ur;
void test_ptr_int_return(void) {
  pr = (ptr_ret_fn)ur;
  ur = (ul_ret_fn)pr;
}

// Pointer-vs-integral parameter types: same size so ABI-compatible.
typedef int (*ptr_param_fn)(void *);
typedef int (*ul_param_fn)(unsigned long);
ptr_param_fn pp;
ul_param_fn up;
void test_ptr_int_param(void) {
  pp = (ptr_param_fn)up;
  up = (ul_param_fn)pp;
}

// Different sizes should still warn.
typedef short (*short_ret_fn)(void);
typedef void *(*ptr_ret_fn2)(void);
short_ret_fn sr;
ptr_ret_fn2 pr2;
void test_ptr_int_diff_size(void) {
  pr2 = (ptr_ret_fn2)sr; /* expected-warning {{cast from 'short_ret_fn' (aka 'short (*)(void)') to 'ptr_ret_fn2' (aka 'void *(*)(void)') converts to incompatible function type}} */
  sr = (short_ret_fn)pr2; /* expected-warning {{cast from 'ptr_ret_fn2' (aka 'void *(*)(void)') to 'short_ret_fn' (aka 'short (*)(void)') converts to incompatible function type}} */
}
