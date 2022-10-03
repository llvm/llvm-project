// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20
// RUN: %clang_cc1 -std=c++2b %s -verify=cxx2b
// RUN: %clang_cc1 -std=c++2b -Wpre-c++2b-compat %s -verify=precxx2b

//cxx2b-no-diagnostics

auto L1 = [] constexpr {};
// cxx20-warning@-1 {{lambda without a parameter clause is a C++2b extension}}
auto L2 = []() static {};
// cxx20-warning@-1 {{static lambdas are a C++2b extension}}
// precxx2b-warning@-2 {{static lambdas are incompatible with C++ standards before C++2b}}
auto L3 = [] static {};
// cxx20-warning@-1 {{lambda without a parameter clause is a C++2b extension}}
// cxx20-warning@-2 {{static lambdas are a C++2b extension}}
// precxx2b-warning@-3 {{static lambdas are incompatible with C++ standards before C++2b}}
