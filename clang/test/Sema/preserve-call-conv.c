// RUN: %clang_cc1 %s -fsyntax-only -triple x86_64-unknown-unknown -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple arm64-unknown-unknown -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple x86_64-unknown-windows-msvc -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple aarch64-unknown-windows-msvc -verify

typedef void typedef_fun_t(int);

void __attribute__((preserve_most)) foo(void *ptr) {
}

void __attribute__((preserve_most(1))) foo1(void *ptr) { // expected-error {{'preserve_most' attribute takes no arguments}}
}

void (__attribute__((preserve_most)) *pfoo1)(void *) = foo;

void (__attribute__((cdecl)) *pfoo2)(void *) = foo; // expected-warning {{incompatible pointer types initializing 'void (*)(void *) __attribute__((cdecl))' with an expression of type 'void (void *) __attribute__((preserve_most))'}}
void (*pfoo3)(void *) = foo; // expected-warning {{incompatible pointer types initializing 'void (*)(void *)' with an expression of type 'void (void *) __attribute__((preserve_most))'}}

typedef_fun_t typedef_fun_foo; // expected-note {{previous declaration is here}}
void __attribute__((preserve_most)) typedef_fun_foo(int x) { } // expected-error {{function declared 'preserve_most' here was previously declared without calling convention}}

struct type_test_foo {} __attribute__((preserve_most));  // expected-warning {{'preserve_most' attribute only applies to functions and methods}}

void __attribute__((preserve_all)) boo(void *ptr) {
}

void __attribute__((preserve_all(1))) boo1(void *ptr) { // expected-error {{'preserve_all' attribute takes no arguments}}
}

void (__attribute__((preserve_all)) *pboo1)(void *) = boo;

void (__attribute__((cdecl)) *pboo2)(void *) = boo; // expected-warning {{incompatible pointer types initializing 'void (*)(void *) __attribute__((cdecl))' with an expression of type 'void (void *) __attribute__((preserve_all))'}}
void (*pboo3)(void *) = boo; // expected-warning {{incompatible pointer types initializing 'void (*)(void *)' with an expression of type 'void (void *) __attribute__((preserve_all))'}}

typedef_fun_t typedef_fun_boo; // expected-note {{previous declaration is here}}
void __attribute__((preserve_all)) typedef_fun_boo(int x) { } // expected-error {{function declared 'preserve_all' here was previously declared without calling convention}}

struct type_test_boo {} __attribute__((preserve_all));  // expected-warning {{'preserve_all' attribute only applies to functions and methods}}
