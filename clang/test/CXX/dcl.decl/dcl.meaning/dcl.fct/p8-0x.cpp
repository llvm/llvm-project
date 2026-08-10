// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

auto f() -> int[32]; // expected-error{{function cannot return array}}
auto g() -> int(int); // expected-error{{function cannot return function}}
auto h() -> auto() -> int; // expected-error{{function cannot return function}}
auto i() -> auto(*)() -> int;
