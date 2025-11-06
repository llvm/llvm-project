// REQUIRES: !system-windows, !system-cygwin

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/layer1.cppm -triple %itanium_abi_triple \
// RUN:     -emit-module-interface -o %t/foo-layer1.pcm
// RUN: %clang_cc1 -std=c++20 %t/layer2.cppm -triple %itanium_abi_triple  \
// RUN:     -emit-module-interface -fmodule-file=foo:layer1=%t/foo-layer1.pcm \
// RUN:     -o %t/foo-layer2.pcm
// RUN: %clang_cc1 -std=c++20 %t/foo-layer1.pcm -emit-llvm -o - | FileCheck %t/layer1.cppm
// RUN: %clang_cc1 -std=c++20 %t/foo-layer2.pcm -emit-llvm -o - \
// RUN:     -fmodule-file=foo:layer1=%t/foo-layer1.pcm | FileCheck %t/layer2.cppm
//
// Check the case about emitting object files from sources directly.
// RUN: %clang_cc1 -std=c++20 %t/layer1.cppm -triple %itanium_abi_triple \
// RUN:     -emit-llvm -o - | FileCheck %t/layer1.cppm
// RUN: %clang_cc1 -std=c++20 %t/layer2.cppm -triple %itanium_abi_triple -emit-llvm  \
// RUN:     -fmodule-file=foo:layer1=%t/foo-layer1.pcm -o - | FileCheck %t/layer2.cppm

//--- layer1.cppm
export module foo:layer1;
struct Fruit {
    virtual ~Fruit() = default;
    virtual void eval();
};

// CHECK-DAG: @_ZTVW3foo5Fruit = unnamed_addr constant
// CHECK-DAG: @_ZTSW3foo5Fruit = constant
// CHECK-DAG: @_ZTIW3foo5Fruit = constant

// Testing that:
// (1) The use of virtual functions won't produce the vtable.
// (2) The definition of key functions won't produce the vtable.
//
//--- layer2.cppm
export module foo:layer2;
import :layer1;
export void layer2_fun() {
    Fruit *b = new Fruit();
    b->eval();
}
void Fruit::eval() {}
// CHECK: @_ZTVW3foo5Fruit = external unnamed_addr constant
// CHECK-NOT: @_ZTSW3foo5Fruit
// CHECK-NOT: @_ZTIW3foo5Fruit
