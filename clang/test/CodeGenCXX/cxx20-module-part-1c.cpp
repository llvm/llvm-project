// RUN: %clang_cc1 -std=c++20 %S/cxx20-module-part-1a.cpp -triple %itanium_abi_triple -emit-module-interface -o %t-inter
// RUN: %clang_cc1 -std=c++20 %S/cxx20-module-part-1b.cpp -triple %itanium_abi_triple -emit-module-interface -o %t-impl
// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -fmodule-file=Foo:inter=%t-inter -fmodule-file=Foo:impl=%t-impl -emit-llvm -o - | FileCheck %s
export module Foo;
export import :inter;
import :impl;

void Wrap() {
  // CHECK: call void @_ZW3Foo4Frobv()
  Frob();
  // CHECK: call void @_ZW3Foo4Quuxv()
  Quux();
}
