// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s

export module func;
class C {
public:
    void member() try {

    } catch (...) {

    }
};

// CHECK: define {{.*}}@_ZNW4func1C6memberEv
