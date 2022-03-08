// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
export module Foo:inter;

// CHECK-DAG: @_ZW3Foo4Frobv(
export void Frob() {
}
