// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++20-extensions -verify=cxx17 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify=cxx20 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++20-extensions -fexperimental-new-constant-interpreter -verify=cxx17 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -fexperimental-new-constant-interpreter -verify=cxx20 %s

union A {
  // cxx20-no-diagnostics
  bool operator==(const A&) const = default; // cxx17-warning {{defaulted comparison operators are a C++20 extension}}
};

A a;
bool b = a == a;
