// RUN: %clang_cc1 -std=c++20 %S/Inputs/cxx20-module-impl-1a.cpp -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s

module Foo;

// CHECK-DAG: @_ZW3Foo8Exportedv(
void Exported() {
}

// CHECK-DAG: @_ZW3Foo6Modulev(
void Module() {
}

// CHECK-DAG: @_ZW3Foo7Module2v(
void Module2() {
}
