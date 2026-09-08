// Check that a key function defined in a TU that includes a -fpch-debuginfo
// PCH still emits the vtable. The PCH object only carries debug info, not the
// vtable definition, so the importing TU must emit it itself.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -fmodules-debuginfo \
// RUN:     -building-pch-with-obj -x c++-header -emit-pch %t/b.h -o %t/b.pch

// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -include-pch %t/b.pch \
// RUN:     -emit-llvm -o - %t/b.cpp | FileCheck %s

// CHECK: @_ZTV1B = {{.*}}constant

//--- b.h
#pragma once
struct B {
    B() = default;
    virtual ~B();
    virtual void f();
};

//--- b.cpp
#include "b.h"
B::~B() { }
void B::f() { }
