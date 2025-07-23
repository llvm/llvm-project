// RUN: %clang_cc1 -fsyntax-only -fsanitize=kcfi -verify %s

#define __cfi_salt(S) __attribute__((cfi_salt(S)))

int foo(int a, int b) __cfi_salt("pepper"); // ok
int foo(int a, int b) __cfi_salt("pepper"); // ok

typedef int (*bar_t)(void) __cfi_salt("pepper"); // ok
typedef int (*bar_t)(void) __cfi_salt("pepper"); // ok

#if 0
// FIXME: These should fail.
int b(void) __cfi_salt("salt 'n") __cfi_salt("pepper");
bar_t bar_fn __cfi_salt("salt 'n");
#endif

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
