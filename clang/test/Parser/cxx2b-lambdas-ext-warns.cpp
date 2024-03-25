// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20
// RUN: %clang_cc1 -std=c++23 %s -verify=cxx23
// RUN: %clang_cc1 -std=c++23 -Wpre-c++23-compat %s -verify=precxx23
// RUN: %clang_cc1 -std=c++23 -pedantic %s -verify=cxx23

//cxx23-no-diagnostics

auto L1 = [] constexpr {};
// cxx20-warning@-1 {{lambda without a parameter clause is a C++23 extension}}
auto L2 = []() static {};
// cxx20-warning@-1 {{static lambdas are a C++23 extension}}
// precxx23-warning@-2 {{static lambdas are incompatible with C++ standards before C++23}}
auto L3 = [] static {};
// cxx20-warning@-1 {{lambda without a parameter clause is a C++23 extension}}
// cxx20-warning@-2 {{static lambdas are a C++23 extension}}
// precxx23-warning@-3 {{static lambdas are incompatible with C++ standards before C++23}}
