// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++1z -emit-pch -o %t.1.ast %S/Inputs/class3.cpp
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++1z -ast-merge %t.1.ast -fsyntax-only -verify %s
// expected-no-diagnostics

class C3 {
  int method_1(C2 *x) {
    return x->x;
  }
};
