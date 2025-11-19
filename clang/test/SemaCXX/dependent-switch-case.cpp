// RUN: %clang_cc1 -std=c++20 %s -verify
// RUN: %clang_cc1 -std=c++20 %s -verify -fexperimental-new-constant-interpreter

constexpr bool e(int){switch(0)0=0:return t(;} // expected-error {{expression is not assignable}} \
                                               // expected-error {{expected 'case' keyword before expression}} \
                                               // expected-error {{expected expression}}
