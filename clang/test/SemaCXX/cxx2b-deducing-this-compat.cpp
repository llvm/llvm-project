// RUN: %clang_cc1 -fsyntax-only -std=c++20 %s -verify

struct S {
    void f(this auto &a); // expected-error {{explicit object parameters are incompatible with C++ standards before C++2b}}
};

void f() {
    (void)[](this auto&a){}; // expected-error {{explicit object parameters are incompatible with C++ standards before C++2b}}
}
