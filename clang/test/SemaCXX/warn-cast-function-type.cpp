// RUN: %clang_cc1 %s -fblocks -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify
// RUN: %clang_cc1 %s -fblocks -fsyntax-only -Wextra -Wno-cast-function-type-strict -verify

int x(long);

typedef int (f1)(long);
typedef int (f2)(void*);
typedef int (f3)(...);
typedef void (f4)(...);
typedef void (f5)(void);
typedef int (f6)(long, int);
typedef int (f7)(long,...);
typedef int (&f8)(long, int);

f1 *a;
f2 *b;
f3 *c;
f4 *d;
f5 *e;
f6 *f;
f7 *g;

struct S
{
  void foo (int*);
  void bar (int);
};

typedef void (S::*mf)(int);

enum E : long;
int efunc(E);

// Produce the underlying `long` type implicitly.
enum E2 { big = __LONG_MAX__ };
int e2func(E2);

void foo() {
  a = (f1 *)x;
  a = (f1 *)efunc; // enum is just type system sugar, still passed as a long.
  a = (f1 *)e2func; // enum is just type system sugar, still passed as a long.
  b = (f2 *)x; // ABI-compatible: long and void* same size on this target.
  b = reinterpret_cast<f2 *>(x); // ABI-compatible: long and void* same size on this target.
  c = (f3 *)x;
  d = (f4 *)x; // expected-warning {{cast from 'int (*)(long)' to 'f4 *' (aka 'void (*)(...)') converts to incompatible function type}}
  e = (f5 *)x;
  f = (f6 *)x; // expected-warning {{cast from 'int (*)(long)' to 'f6 *' (aka 'int (*)(long, int)') converts to incompatible function type}}
  g = (f7 *)x;

  mf p1 = (mf)&S::foo; // ABI-compatible: int* and int same size on this target.

  f8 f2 = (f8)x; // expected-warning {{cast from 'int (long)' to 'f8' (aka 'int (&)(long, int)') converts to incompatible function type}}
  (void)f2;

  int (^y)(long);
  f = (f6 *)y; // expected-warning {{cast from 'int (^)(long)' to 'f6 *' (aka 'int (*)(long, int)') converts to incompatible function type}}
}

// Pointer-vs-integral return types: same size so ABI-compatible on this target.
typedef void *(*ptr_ret_fn)(void);
typedef unsigned long (*ul_ret_fn)(void);
ptr_ret_fn pr;
ul_ret_fn ur;
void test_ptr_int_return() {
  pr = (ptr_ret_fn)ur;
  ur = (ul_ret_fn)pr;
}

// Pointer-vs-integral with reinterpret_cast.
void test_ptr_int_reinterpret_cast() {
  pr = reinterpret_cast<ptr_ret_fn>(ur);
  ur = reinterpret_cast<ul_ret_fn>(pr);
}

// Different sizes should still warn.
typedef short (*short_ret_fn)(void);
short_ret_fn sr;
void test_ptr_int_diff_size() {
  void *(*pr3)(void);
  pr3 = (void *(*)(void))sr; // expected-warning {{cast from 'short_ret_fn' (aka 'short (*)()') to 'void *(*)()' converts to incompatible function type}}
}
