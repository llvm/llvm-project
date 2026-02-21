// RUN: %clang_cc1 %s -fsyntax-only -Wcast-function-type -Wno-cast-function-type-strict -verify
// RUN: %clang_cc1 %s -fsyntax-only -Wextra -Wno-cast-function-type-strict -verify

int x(long);
int y(short);

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
  b = (f2 *)y; /* expected-warning {{cast from 'int (*)(short)' to 'f2 *' (aka 'int (*)(void *)') converts to incompatible function type}} */
  c = (f3 *)x;
  d = (f4 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f4 *' (aka 'void (*)()') converts to incompatible function type}} */
  e = (f5 *)x;
  f = (f6 *)x; /* expected-warning {{cast from 'int (*)(long)' to 'f6 *' (aka 'int (*)(long, int)') converts to incompatible function type}} */
  g = (f7 *)x;
}

// Test pointer-integer conversions with same width (issue #178388)
// __INTPTR_TYPE__ and __UINTPTR_TYPE__ are guaranteed to have the same size as pointers.
typedef __UINTPTR_TYPE__ uintptr_t;
typedef __INTPTR_TYPE__ intptr_t;

void *returns_ptr(void);
uintptr_t returns_uintptr(void);
intptr_t returns_intptr(void);

typedef void *(*fn_returning_ptr)(void);
typedef uintptr_t (*fn_returning_uintptr)(void);
typedef intptr_t (*fn_returning_intptr)(void);

int takes_ptr(void *);
int takes_uintptr(uintptr_t);
int takes_intptr(intptr_t);

typedef int (*fn_taking_ptr)(void *);
typedef int (*fn_taking_uintptr)(uintptr_t);
typedef int (*fn_taking_intptr)(intptr_t);

void test_ptr_int_same_width(void) {
  // Return type: pointer to same-width integer should not warn
  fn_returning_uintptr p1 = (fn_returning_uintptr)returns_ptr;    // no warning
  fn_returning_intptr p2 = (fn_returning_intptr)returns_ptr;      // no warning
  fn_returning_ptr p3 = (fn_returning_ptr)returns_uintptr;        // no warning
  fn_returning_ptr p4 = (fn_returning_ptr)returns_intptr;         // no warning

  // Parameter type: pointer to same-width integer should not warn
  fn_taking_uintptr p5 = (fn_taking_uintptr)takes_ptr;            // no warning
  fn_taking_intptr p6 = (fn_taking_intptr)takes_ptr;              // no warning
  fn_taking_ptr p7 = (fn_taking_ptr)takes_uintptr;                // no warning
  fn_taking_ptr p8 = (fn_taking_ptr)takes_intptr;                 // no warning
}
