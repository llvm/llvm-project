// RUN: %clang_cc1 %s -verify -fsyntax-only -Wc++2a-compat -std=c++17

#define concept constexpr bool
template<typename T>
concept x = 0;
#undef concept

int concept = 0; // expected-warning {{'concept' is a keyword in C++2a}}
int requires = 0; // expected-warning {{'requires' is a keyword in C++2a}}
