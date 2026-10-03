// RUN: %clang_cc1  -fsyntax-only -verify %s

// Function annotations.
[[clang::unsafe_buffer_usage]]
void f(int *buf, int size);
void g(int *buffer [[clang::unsafe_buffer_usage]], int size); // expected-warning {{'clang::unsafe_buffer_usage' attribute only applies to functions}}
void h(int *buffer [[clang::unsafe_buffer_usage("buffer")]], int size); // expected-error {{'clang::unsafe_buffer_usage' attribute takes no arguments}}
