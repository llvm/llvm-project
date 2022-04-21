// RUN: %clang_cc1 -triple i386-pc-unknown -fsyntax-only -Wstrict-prototypes -Wno-implicit-function-declaration -verify %s
// RUN: %clang_cc1 -triple i386-pc-unknown -fsyntax-only -Wstrict-prototypes -Wno-implicit-function-declaration -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// function definition with 0 params, no prototype, no preceding declaration.
void foo0() {} // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}

// function declaration with unspecified params
void foo1(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
             // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:11}:"void"
// function declaration with 0 params
void foo2(void);

// function definition with 0 params, no prototype.
void foo1() {} // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
// function definition with 0 params, prototype.
void foo2(void) {}

// function type typedef unspecified params
typedef void foo3(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
                     // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:19-[[@LINE-1]]:19}:"void"

// global fp unspecified params
void (*foo4)(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
                // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"void"

// struct member fp unspecified params
struct { void (*foo5)(); } s; // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
                              // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:23-[[@LINE-1]]:23}:"void"

// param fp unspecified params
void bar2(void (*foo6)()) { // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
                            // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:24-[[@LINE-1]]:24}:"void"
  // local fp unspecified params
  void (*foo7)() = 0; // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
                      // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"void"
  // array fp unspecified params
  void (*foo8[2])() = {0}; // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
                           // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:19-[[@LINE-1]]:19}:"void"
}

// function type cast using using an anonymous function declaration
void bar3(void) {
  // casting function w/out prototype to unspecified params function type
  (void)(void(*)()) foo1; // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
                          // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:18-[[@LINE-1]]:18}:"void"
  // .. specified params
  (void)(void(*)(void)) foo1;
}

// K&R function definition not preceded by full prototype
int foo9(a, b) // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}
  int a, b;
{
  return a + b;
}

// Function declaration with no types
void foo10(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} \
                 expected-note {{a function declaration without a prototype is not supported in C2x}}
              // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"void"
// K&R function definition with incomplete param list declared
void foo10(p, p2) void *p; {} // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

void foo11(int p, int p2);
void foo11(p, p2) int p; int p2; {} // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}

// PR31020
void __attribute__((cdecl)) foo12(d) // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}
  short d;
{}

// No warnings for variadic functions. Overloadable attribute is required
// to avoid err_ellipsis_first_param error.
// rdar://problem/33251668
void foo13(...) __attribute__((overloadable));
void foo13(...) __attribute__((overloadable)) {}

// We should not generate a strict-prototype warning for an implicit
// declaration.  Leave that up to the implicit-function-declaration warning.
void foo14(void) {
  foo14_call(); // no-warning
}

// Ensure that redeclarations involving a typedef type work properly, even if
// there are function attributes involved in the declaration.
typedef void foo_t(unsigned val);
__attribute__((noreturn)) foo_t foo15;
foo_t foo15; // OK
void foo15(unsigned val); // OK

foo_t foo16;
void foo16(unsigned val); // OK
