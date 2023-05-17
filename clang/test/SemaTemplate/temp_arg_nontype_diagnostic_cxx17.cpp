// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wpre-c++17-compat %s

template <decltype(auto) n> // expected-warning {{non-type template parameters declared with 'decltype(auto)' are incompatible with C++ standards before C++17}}
struct B{};

template <auto n> // expected-warning {{non-type template parameters declared with 'auto' are incompatible with C++ standards before C++17}}
struct A{};
