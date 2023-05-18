// RUN: %clang_cc1  -fsyntax-only -verify %s

// Function annotations.
[[clang::unsafe_buffer_usage]]
void f(int *buf, int size);
void g(int *buffer [[clang::unsafe_buffer_usage("buffer")]], int size); // expected-warning {{'unsafe_buffer_usage' attribute only applies to functions}}
