// RUN: %clang_cc1 -std=c++20 %S/cxx20-module-sub-1a.cppm -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s

export module FOO.BAZ;
import FOO.BAR;

namespace Bob {

// CHECK-DAG: void @_ZN3BobW3FOOW3BAZ3FooEPS0_W3BAR1APNS_S2_1BE(
void Foo (A *, B*) {
}
}

// CHECK-DAG: void @_ZW3FOOW3BAZ3BarPS_W3BAR1APN3BobS1_1BE(
void Bar (A *, Bob::B*) {
}
