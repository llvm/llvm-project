// RUN: %clang_cc1  -fsyntax-only -verify %s

// Decl annotations.
void f(int *a __attribute__((acquire_handle("Fuchsia"))));
void (*fp)(int handle [[clang::use_handle("Fuchsia")]]);
auto lambda = [](int handle [[clang::use_handle("Fuchsia")]]){};
void g(int a __attribute__((acquire_handle("Fuchsia")))); // expected-error {{attribute only applies to output parameters}}
void h(int *a __attribute__((acquire_handle))); // expected-error {{'acquire_handle' attribute takes one argument}}
void h(int *a __attribute__((acquire_handle(1)))); // expected-error {{expected string literal as argument of 'acquire_handle' attribute}}
void h(int *a __attribute__((acquire_handle("RandomString", "AndAnother")))); // expected-error {{'acquire_handle' attribute takes one argument}}
__attribute__((release_handle("Fuchsia"))) int i(); // expected-warning {{'release_handle' attribute only applies to parameters}}
__attribute__((use_handle("Fuchsia"))) int j(); // expected-warning {{'use_handle' attribute only applies to parameters}}
int a __attribute__((acquire_handle("Fuchsia"))); // expected-warning {{'acquire_handle' attribute only applies to functions, typedefs, and parameters}}
int (* __attribute__((acquire_handle("Fuchsia"))) fpt)(char *); // expected-warning {{'acquire_handle' attribute only applies to functions, typedefs, and parameters}}

// Type annotations.
auto lambdat = [](int handle __attribute__((use_handle("Fuchsia"))))
    __attribute__((acquire_handle("Fuchsia"))) -> int { return 0; };
int __attribute((acquire_handle("Fuchsia"))) ta; // expected-warning {{'acquire_handle' attribute only applies to functions, typedefs, and parameters}}
int open(const char *path, int flags, ...) [[clang::acquire_handle]]; // expected-error {{'acquire_handle' attribute takes one argument}}

// Typedefs.
typedef int callback(char *) __attribute__((acquire_handle("Fuchsia")));
