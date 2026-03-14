// RUN: %clang_cc1 %s -verify -fsyntax-only -Wc++20-compat -std=c++17

#define concept constexpr bool
template<typename T>
concept x = 0;
#undef concept

int co_await = 0; // expected-warning {{'co_await' is a keyword in C++20}}
int co_return = 0; // expected-warning {{'co_return' is a keyword in C++20}}
int co_yield = 0; // expected-warning {{'co_yield' is a keyword in C++20}}
int char8_t = 0; // expected-warning {{'char8_t' is a keyword in C++20}}
int concept = 0; // expected-warning {{'concept' is a keyword in C++20}}
int requires = 0; // expected-warning {{'requires' is a keyword in C++20}}
int consteval = 0; // expected-warning {{'consteval' is a keyword in C++20}}
int constinit = 0; // expected-warning {{'constinit' is a keyword in C++20}}
