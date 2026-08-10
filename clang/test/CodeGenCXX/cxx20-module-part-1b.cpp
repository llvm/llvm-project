// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
export module Foo:impl;

// CHECK-DAG: @_ZW3Foo4Quuxv(
export void Quux() {
}
