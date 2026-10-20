// RUN: %clang_cc1 -std=c11 -fsyntax-only -fsanitize=kcfi -verify %s
// RUN: %clang_cc1 -std=c89 -DKNR -fsyntax-only -fsanitize=kcfi -verify %s

#define __cfi_salt(S) __attribute__((cfi_salt(S)))

int bad1(void) __cfi_salt(); // expected-error{{'cfi_salt' attribute takes one argument}}
int bad2(void) __cfi_salt(42); // expected-error{{expected string literal as argument of 'cfi_salt' attribute}}
int bad3(void) __attribute__((cfi_salt("a", "b", "c"))); // expected-error{{'cfi_salt' attribute takes one argument}}


int foo(int a, int b) __cfi_salt("pepper"); // ok
int foo(int a, int b) __cfi_salt("pepper"); // ok

#ifndef KNR
typedef int (*bar_t)(void) __cfi_salt("pepper"); // ok
typedef int (*bar_t)(void) __cfi_salt("pepper"); // ok
#endif

// FIXME: Should we allow this?
// int b(void) __cfi_salt("salt 'n") __cfi_salt("pepper");
// bar_t bar_fn __cfi_salt("salt 'n");

int baz __cfi_salt("salt"); // expected-warning{{'cfi_salt' only applies to function types}}

int baz_fn(int a, int b) __cfi_salt("salt 'n"); // expected-note{{previous declaration is here}}
int baz_fn(int a, int b) __cfi_salt("pepper"); // expected-error{{conflicting types for 'baz_fn'}}

int mux_fn(int a, int b) __cfi_salt("salt 'n"); // expected-note{{previous declaration is here}}
int mux_fn(int a, int b) __cfi_salt("pepper") { // expected-error{{conflicting types for 'mux_fn'}}
  return a * b;
}

typedef int qux_t __cfi_salt("salt"); // expected-warning{{'cfi_salt' only applies to function types}}

typedef int (*quux_t)(void) __cfi_salt("salt 'n"); // expected-note{{previous definition is here}}
typedef int (*quux_t)(void) __cfi_salt("pepper"); // expected-error{{typedef redefinition with different type}}

void func1(int a) __cfi_salt("pepper"); // expected-note{{previous declaration is here}}
void func1(int a) { } // expected-error{{conflicting types for 'func1'}}
void (*fp1)(int) = func1; // expected-error{{incompatible function pointer types initializing 'void (*)(int)' with an expression of type 'void (int)'}}

void func2(int) [[clang::cfi_salt("test")]]; // expected-note{{previous declaration is here}}
void func2(int a) { } // expected-error{{conflicting types for 'func2'}}
void (*fp2)(int) = func2; // expected-error{{incompatible function pointer types initializing 'void (*)(int)' with an expression of type 'void (int)'}}

void func3(int) __cfi_salt("pepper"); // ok
void func3(int a) __cfi_salt("pepper") { } // ok
void (* __cfi_salt("pepper") fp3)(int) = func3; // ok
void (*fp3_noattr)(int) = func3; // expected-error{{incompatible function pointer types initializing 'void (*)(int)' with an expression of type 'void (int)'}}

void func4(int) [[clang::cfi_salt("test")]]; // ok
void func4(int a) [[clang::cfi_salt("test")]] { } // ok
void (* [[clang::cfi_salt("test")]] fp4)(int) = func4; // ok
void (*fp4_noattr)(int) = func4; // expected-error{{incompatible function pointer types initializing 'void (*)(int)' with an expression of type 'void (int)'}}

#ifdef KNR
// K&R C function without a prototype
void func() __attribute__((cfi_salt("pepper"))); // expected-error {{attribute only applies to non-K&R-style functions}}
void (*fp)() __attribute__((cfi_salt("pepper")));  // expected-error {{attribute only applies to non-K&R-style functions}}
#endif
