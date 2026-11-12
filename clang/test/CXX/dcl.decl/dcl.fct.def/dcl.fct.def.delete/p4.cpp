// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename> void func();
template<> void func<int>() = delete;

template<typename> void func2();
template<> void func2<int>(); // expected-note {{previous declaration is here}}
template<> void func2<int>() = delete; // expected-error {{deleted definition must be first declaration}}
