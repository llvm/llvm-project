// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx23 -std=c++23 -Wpre-c++23-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

void looks_like_decltype_auto() {
  decltype(auto(42)) b = 42; // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}} \
                                cxx23-warning {{'auto' as a functional-style cast is incompatible with C++ standards before C++23}}
  decltype(long *) a = 42;   // expected-error {{expected '(' for function-style cast or type construction}} \
                                expected-error {{expected expression}}
  decltype(auto *) a = 42;   // expected-error {{expected '(' for function-style cast or type construction}} \
                                expected-error {{expected expression}}
  decltype(auto()) c = 42;   // expected-error {{initializer for functional-style cast to 'auto' is empty}} \
                                cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}} \
                                cxx23-warning {{'auto' as a functional-style cast is incompatible with C++ standards before C++23}}
}

struct looks_like_declaration {
  int n;
} a;

using T = looks_like_declaration *;
void f() { T(&a)->n = 1; }
void g() { auto(&a)->n = 0; } // cxx23-warning {{before C++23}} \
                              // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
void h() { auto{&a}->n = 0; } // cxx23-warning {{before C++23}} \
                              // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

void e(auto (*p)(int y) -> decltype(y)) {}

struct M;
struct S{
    S operator()();
    S* operator->();
    int N;
    int M;
} s; // expected-note {{here}}

void test() {
    auto(s)()->N; // expected-warning {{expression result unused}} \
                  // cxx23-warning {{before C++23}} \
                  // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
    auto(s)()->M; // expected-error {{redefinition of 's' as different kind of symbol}}
}

void test_paren() {
    int a = (auto(0)); // cxx23-warning {{before C++23}} \
                       // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
    int b = (auto{0}); // cxx23-warning {{before C++23}} \
                       // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
}
