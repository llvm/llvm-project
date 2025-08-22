// RUN: %clang_cc1 -std=c++2c -fsyntax-only -fblocks -verify %s

void a(...);

void b(auto...);
void c(auto, ...);

void d(auto......); // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}} \
                    // expected-warning {{'...' in this location creates a C-style varargs function}} \
                    // expected-note {{preceding '...' declares a function parameter pack}} \
                    // expected-note {{insert ',' before '...' to silence this warning}}
void e(auto..., ...);

void f(auto x...); // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}
void g(auto x, ...);

void h(auto... x...); // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}} \
                      // expected-warning {{'...' in this location creates a C-style varargs function}} \
                      // expected-note {{preceding '...' declares a function parameter pack}} \
                      // expected-note {{insert ',' before '...' to silence this warning}}
void i(auto... x, ...);

template<class ...Ts>
void j(Ts... t...) {}; // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}} \
                       // expected-warning {{'...' in this location creates a C-style varargs function}} \
                       // expected-note {{preceding '...' declares a function parameter pack}} \
                       // expected-note {{insert ',' before '...' to silence this warning}}
template<class ...Ts>
void k(Ts... t, ...) {}

void l(int...); // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}
void m(int, ...);

void n(int x...); // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}
void o(int x, ...);

struct S {
  void p(this S...) {} // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}
  void f(int = {}...); // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}
};

template<class ...Ts>
void q(Ts......) {} // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}} \
                    // expected-warning {{'...' in this location creates a C-style varargs function}} \
                    // expected-note {{preceding '...' declares a function parameter pack}} \
                    // expected-note {{insert ',' before '...' to silence this warning}}

template<class T>
void r(T...) {} // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}

auto type_specifier = (void (*)(int...)) nullptr; // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}

auto lambda = [](int...) {}; // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}

auto block = ^(int...){}; // expected-warning {{declaration of a variadic function without a comma before '...' is deprecated}}
