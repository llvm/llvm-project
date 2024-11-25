// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

void a(...);

void b(auto...);
void c(auto, ...);

void d(auto......); // expected-warning {{variadic parameters that are not preceded by a comma are deprecated}} \
                    // expected-warning {{'...' in this location creates a C-style varargs function}} \
                    // expected-note {{preceding '...' declares a function parameter pack}} \
                    // expected-note {{insert ',' before '...' to silence this warning}}
void e(auto..., ...);

void f(auto x...); // expected-warning {{variadic parameters that are not preceded by a comma are deprecated}}
void g(auto x, ...);

void h(auto... x...); // expected-warning {{variadic parameters that are not preceded by a comma are deprecated}} \
                      // expected-warning {{'...' in this location creates a C-style varargs function}} \
                      // expected-note {{preceding '...' declares a function parameter pack}} \
                      // expected-note {{insert ',' before '...' to silence this warning}}
void i(auto... x, ...);

template<class ...T>
void j(T... t...); // expected-warning {{variadic parameters that are not preceded by a comma are deprecated}} \
                   // expected-warning {{'...' in this location creates a C-style varargs function}} \
                   // expected-note {{preceding '...' declares a function parameter pack}} \
                   // expected-note {{insert ',' before '...' to silence this warning}}
template<class ...T>
void k(T... t, ...);

void l(int...); // expected-warning {{variadic parameters that are not preceded by a comma are deprecated}}
void m(int, ...);

void n(int x...); // expected-warning {{variadic parameters that are not preceded by a comma are deprecated}}
void o(int x, ...);

