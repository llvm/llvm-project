// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify=cxx17 %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=pre-cxx20-compat -Wpre-c++20-compat %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=cxx20-compat -Wc++20-compat %s
// cxx20-compat-no-diagnostics

// cxx17-error@+4 {{unknown type name 'consteval'; did you mean 'constexpr'}}
// cxx17-warning@+3 {{missing 'typename' prior to dependent type name 'T::type' is a C++20 extension}}
// pre-cxx20-compat-warning@+2 {{'consteval' specifier is incompatible with C++ standards before C++20}}
// pre-cxx20-compat-warning@+1 {{missing 'typename' prior to dependent type name 'T::type' is incompatible with C++ standards before C++20}}
template<typename T> consteval T::type f();

// cxx17-error@+2 {{unknown type name 'constinit'}}
// pre-cxx20-compat-warning@+1 {{'constinit' specifier is incompatible with C++ standards before C++20}}
constinit int x = 4;
