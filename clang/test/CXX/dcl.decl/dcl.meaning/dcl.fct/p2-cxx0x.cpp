// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s 

auto a() -> int; // ok
const auto b() -> int; // expected-error {{function with trailing return type must specify return type 'auto', not 'auto const'}}
auto *c() -> int; // expected-error {{function with trailing return type must specify return type 'auto', not 'auto *'}}
auto (d() -> int); // expected-error {{trailing return type may not be nested within parentheses}}
auto e() -> auto (*)() -> auto (*)() -> void; // ok: same as void (*(*e())())();
