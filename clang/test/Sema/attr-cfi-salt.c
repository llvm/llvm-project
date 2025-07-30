// RUN: %clang_cc1 -fsyntax-only -fsanitize=kcfi -verify %s

#define __cfi_salt(S) __attribute__((cfi_salt(S)))

int bad1() __cfi_salt();
    // expected-error@-1{{'cfi_salt' attribute takes one argument}}
int bad2() __cfi_salt(42);
    // expected-error@-1{{expected string literal as argument of 'cfi_salt' attribute}}

int foo(int a, int b) __cfi_salt("pepper"); // ok
int foo(int a, int b) __cfi_salt("pepper"); // ok

typedef int (*bar_t)(void) __cfi_salt("pepper"); // ok
typedef int (*bar_t)(void) __cfi_salt("pepper"); // ok

// FIXME: Should we allow this?
// int b(void) __cfi_salt("salt 'n") __cfi_salt("pepper");
// bar_t bar_fn __cfi_salt("salt 'n");

int baz __cfi_salt("salt");
    // expected-warning@-1{{'cfi_salt' only applies to function types}}

int baz_fn(int a, int b) __cfi_salt("salt 'n");
    // expected-note@-1{{previous declaration is here}}
int baz_fn(int a, int b) __cfi_salt("pepper");
    // expected-error@-1{{conflicting types for 'baz_fn'}}

int mux_fn(int a, int b) __cfi_salt("salt 'n");
    // expected-note@-1{{previous declaration is here}}
int mux_fn(int a, int b) __cfi_salt("pepper") {
    // expected-error@-1{{conflicting types for 'mux_fn'}}
  return a * b;
}

typedef int qux_t __cfi_salt("salt");
    // expected-warning@-1{{'cfi_salt' only applies to function types}}

typedef int (*quux_t)(void) __cfi_salt("salt 'n");
    // expected-note@-1{{previous definition is here}}
typedef int (*quux_t)(void) __cfi_salt("pepper");
    // expected-error@-1{{typedef redefinition with different type}}

void func(int a) __cfi_salt("pepper");
    // expected-note@-1{{previous declaration is here}}
void func(int a) { }
    // expected-error@-1{{conflicting types for 'func'}}

void func2(int) [[clang::cfi_salt("test")]];
void func2(int a) { }
void (*fp)(int) = func2;
