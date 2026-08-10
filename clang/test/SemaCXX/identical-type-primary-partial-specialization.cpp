// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

template <decltype(auto) a>
struct S { // expected-note {{previous definition is here}}
    static constexpr int i = 42;
};

template <decltype(auto) a>
struct S<a> { // expected-error {{class template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}} \
              // expected-error {{redefinition of 'S'}}
    static constexpr int i = 0;
};
