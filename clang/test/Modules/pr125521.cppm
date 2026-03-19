// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod2.cppm -emit-module-interface -o %t/mod2.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-module-interface -o %t/mod1.pcm \
// RUN:     -fmodule-file=Mod2=%t/mod2.pcm
// RUN: %clang_cc1 -std=c++20 %t/test.cc -fmodule-file=Mod2=%t/mod2.pcm -fmodule-file=Mod=%t/mod1.pcm \
// RUN:     -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/mod2.cppm -emit-module-interface -o %t/mod2.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-module-interface -o %t/mod1.pcm \
// RUN:     -fmodule-file=Mod2=%t/mod2.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod1.pcm  -fmodule-file=Mod2=%t/mod2.pcm -emit-llvm -o - \
// RUN:     | FileCheck %t/mod1.cppm

//--- hello.h
template <typename V> int get() noexcept {return 0;};

template <typename T>
class List
{
    template <typename V> friend int get() noexcept;
};

//--- mod2.cppm
module;
#include "hello.h"
export module Mod2;
export const char *modFn2() {
    List<int> a;
    return "hello";
}

//--- mod1.cppm
module;
#include "hello.h"
export module Mod;
import Mod2;
export extern "C" const char *modFn() {
    List<int> a;
    List<double> b;
    return modFn2();
}

// Fine enough to check it won't crash.
// CHECK: define {{.*}}@modFn

//--- test.cc
// expected-no-diagnostics
import Mod;
import Mod2;

void test() {
    modFn();
    modFn2();
}
