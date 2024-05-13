// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx23 -std=c++23 -Wpre-c++23-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

void looks_like_decltype_auto() {
  decltype(auto(42)) b = 42; // cxx20-error {{'auto' not allowed here}} \
                                cxx23-warning {{'auto' as a functional-style cast is incompatible with C++ standards before C++23}}
  decltype(long *) a = 42;   // expected-error {{expected '(' for function-style cast or type construction}} \
                                expected-error {{expected expression}}
  decltype(auto *) a = 42;   // expected-error {{expected '(' for function-style cast or type construction}} \
                                expected-error {{expected expression}}
  decltype(auto()) c = 42;   // cxx23-error {{initializer for functional-style cast to 'auto' is empty}} \
                                cxx20-error {{'auto' not allowed here}}
}

struct looks_like_declaration {
  int n;
} a;

using T = looks_like_declaration *;
void f() { T(&a)->n = 1; }
void g() { auto(&a)->n = 0; } // cxx23-warning {{before C++23}} \
                              // cxx20-error {{declaration of variable 'a' with deduced type 'auto (&)' requires an initializer}} \
                              // cxx20-error {{expected ';' at end of declaration}}
void h() { auto{&a}->n = 0; } // cxx23-warning {{before C++23}} \
                              // cxx20-error {{expected unqualified-id}} \
                              // cxx20-error {{expected expression}}

void e(auto (*p)(int y) -> decltype(y)) {}

struct M;
struct S{
    S operator()();
    S* operator->();
    int N;
    int M;
} s; // expected-note {{here}}

void test() {
    auto(s)()->N; // cxx23-warning {{expression result unused}} \
                  // cxx23-warning {{before C++23}} \
                  // cxx20-error {{unknown type name 'N'}}
    auto(s)()->M; // expected-error {{redefinition of 's' as different kind of symbol}}
}

void test_paren() {
    int a = (auto(0)); // cxx23-warning {{before C++23}} \
                       // cxx20-error {{expected expression}} \
                       // cxx20-error {{expected ')'}} \
                       // cxx20-note  {{to match this '('}}
    int b = (auto{0}); // cxx23-warning {{before C++23}} \
                       // cxx20-error {{expected expression}} \
                       // cxx20-error {{expected ')'}} \
                       // cxx20-note  {{to match this '('}}
}
