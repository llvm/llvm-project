// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx2b -std=c++2b -Wpre-c++2b-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

void looks_like_decltype_auto() {
  decltype(auto(42)) b = 42; // cxx20-error {{'auto' not allowed here}} \
                                cxx2b-warning {{'auto' as a functional-style cast is incompatible with C++ standards before C++2b}}
  decltype(long *) a = 42;   // expected-error {{expected '(' for function-style cast or type construction}} \
                                expected-error {{expected expression}}
  decltype(auto *) a = 42;   // expected-error {{expected '(' for function-style cast or type construction}} \
                                expected-error {{expected expression}}
  decltype(auto()) c = 42;   // cxx2b-error {{initializer for functional-style cast to 'auto' is empty}} \
                                cxx20-error {{'auto' not allowed here}}
}

struct looks_like_declaration {
  int n;
} a;

using T = looks_like_declaration *;
void f() { T(&a)->n = 1; }
// FIXME: They should be deemed expressions without breaking function pointer
//        parameter declarations with trailing return types.
// void g() { auto(&a)->n = 0; }
// void h() { auto{&a}->n = 0; }
