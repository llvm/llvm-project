// RUN: %clang_cc1 -std=c++2b %s -verify
// RUN: %clang_cc1 -std=c++2a %s -verify

auto L0 = []<> { }; //expected-error {{cannot be empty}}

auto L1 = []<typename T1, typename T2> { };
auto L2 = []<typename T1, typename T2>(T1 arg1, T2 arg2) -> T1 { };
auto L3 = []<typename T>(auto arg) { T t; };
auto L4 = []<int I>() { };
