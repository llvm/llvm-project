// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

template <template <typename> class C> struct S { // expected-note {{template is declared here}}
    friend C(); //expected-error {{cannot specify deduction guide for template template parameter 'C'}}
};
