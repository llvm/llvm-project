// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/a.cpp -fmodule-file=a=%t/a.pcm -emit-llvm -o - | FileCheck %t/a.cpp

//--- a.cppm
export module a;
int func();
static int a = func();

//--- a.cpp
import a;

// CHECK-NOT: internal global
// CHECK-NOT: __cxx_global_var_init

