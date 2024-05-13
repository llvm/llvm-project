// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

// module-purview extern "C++" semantics not implemented
// XFAIL: *

export module FOO;
extern "C++" export class A;
export class B;

// CHECK-DAG: void @_ZW3FOO3FooP1APNS_1B(
export void Foo (A *, B*) {
}

extern "C++" {
// CHECK-DAG: void @_Z3BarP1APW3FOO1B(
export void Bar (A *, B*) {
}
}
