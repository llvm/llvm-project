// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++1z -emit-pch -o %t.1.ast %S/Inputs/init-ctors-classes.cpp
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++1z -ast-merge %t.1.ast -fsyntax-only -verify %s
// expected-no-diagnostics

class B {
  int method_1() {
    A a(0);
    return a.x;
  }
};
