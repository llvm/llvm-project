// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/m.cppm -emit-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/use.cpp -fmodule-file=m=%t/m.pcm -emit-llvm -o - | FileCheck %t/use.cpp
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/use.cpp -fmodule-file=m=%t/m.pcm -emit-llvm -o - | FileCheck %t/use.cpp

//--- header.h
#pragma once

template <unsigned T>
class Templ {
public:
    void lock() { __set_locked_bit(); }

private:
    static constexpr auto __set_locked_bit = [](){};
};

class JT {
public:
    ~JT() {
        Templ<4> state;
        state.lock();
    }
};

//--- m.cppm
module;
#include "header.h"
export module m;
export struct M {
    JT jt;
};
//--- use.cpp
#include "header.h"
import m;

int main() {
    M m;
    return 0;
}

// CHECK: @_ZN5TemplILj4EE16__set_locked_bitE = {{.*}}linkonce_odr
