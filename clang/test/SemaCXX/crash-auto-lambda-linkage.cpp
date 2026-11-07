// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// https://github.com/llvm/llvm-project/issues/22203
auto g = [] { virtual g() // expected-error {{a type specifier is required for all declarations}} expected-warning {{empty parentheses interpreted as a function declaration}} expected-note {{replace parentheses with an initializer to declare a variable}} expected-error {{expected ';' at end of declaration}} expected-error {{expected '}'}} expected-note {{to match this '{'}} expected-error {{expected ';' after top level declarator}}
