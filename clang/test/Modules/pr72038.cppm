// RUN: %clang_cc1 -std=c++20 %s -fsyntax-only -verify

// expected-no-diagnostics
module;
struct kevent { };
void kevent(int x);
export module my_mod;
struct kevent evt;
