// RUN: %clang_cc1 %s -fsyntax-only -triple x86_64-unknown-unknown -verify

typedef void typedef_fun_t(int);

void __attribute__((preserve_none)) boo(void *ptr) {
}

void __attribute__((preserve_none(1))) boo1(void *ptr) { // expected-error {{'preserve_none' attribute takes no arguments}}
}

void (__attribute__((preserve_none)) *pboo1)(void *) = boo;

void (__attribute__((cdecl)) *pboo2)(void *) = boo; // expected-error {{incompatible function pointer types initializing 'void (*)(void *) __attribute__((cdecl))' with an expression of type 'void (void *) __attribute__((preserve_none))'}}
void (*pboo3)(void *) = boo; // expected-error {{incompatible function pointer types initializing 'void (*)(void *)' with an expression of type 'void (void *) __attribute__((preserve_none))'}}

typedef_fun_t typedef_fun_boo; // expected-note {{previous declaration is here}}
void __attribute__((preserve_none)) typedef_fun_boo(int x) { } // expected-error {{function declared 'preserve_none' here was previously declared without calling convention}}

struct type_test_boo {} __attribute__((preserve_none));  // expected-warning {{'preserve_none' attribute only applies to functions and function pointers}}
