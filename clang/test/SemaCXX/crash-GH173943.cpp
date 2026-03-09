// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++03 %s

// https://github.com/llvm/llvm-project/issues/173943

constexpr void f(this auto& self) // expected-note {{candidate function}}
    // expected-error@-1 {{unknown type name 'constexpr'}}
    // expected-error@-2 {{'auto' not allowed in function prototype}}
    // expected-error@-3 {{explicit object parameters are incompatible with C++ standards before C++2b}}
    // expected-error@-4 {{expected ';' after top level declarator}}
    // expected-error@-5 {{an explicit object parameter cannot appear in a non-member function}}
    // expected-warning@-6 {{'auto' type specifier is a C++11 extension}}

void g() {
  f(); // expected-error {{no matching function for call to 'f'}}
}
