// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -std=c++17 %s

class A{};

auto f() {
  return typeid(A);
}

// CHECK: error: you need to include <typeinfo> or import std before using the 'typeid' operator