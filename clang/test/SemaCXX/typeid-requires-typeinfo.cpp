// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify=cxx20 -std=c++20 %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify=cxx17 -std=c++17 %s

class A{};

auto f() {
  // cxx20-error@+2 {{you need to include <typeinfo> or import std before using the 'typeid' operator}}
  // cxx17-error@+1 {{you need to include <typeinfo> before using the 'typeid' operator}}
  return typeid(A);
}
