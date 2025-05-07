// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify=cxx17 %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=cxx20 -Wpre-c++20-compat %s

// cxx17-error@+3 {{unknown type name 'consteval'; did you mean 'constexpr'}}
// cxx17-warning@+2 {{missing 'typename' prior to dependent type name 'T::type' is a C++20 extension}}
// cxx20-warning@+1 {{missing 'typename' prior to dependent type name 'T::type' is incompatible with C++ standards before C++20}}
template<typename T> consteval T::type f();
