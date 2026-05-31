// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++03 %s

// https://github.com/llvm/llvm-project/issues/173943

constexpr void f(this auto& self)
    // expected-error@-1 {{unknown type name 'constexpr'}}
    // expected-warning@-2 {{'auto' parameters are a C++20 extension}}
    // expected-error@-3 {{expected ';' after top level declarator}}
    // expected-error@-4 {{an explicit object parameter cannot appear in a non-member function}}
    // expected-warning@-5 {{'auto' type specifier is a C++11 extension}}

void g() {
  f(); // expected-error {{no matching function for call to 'f'}}
}
