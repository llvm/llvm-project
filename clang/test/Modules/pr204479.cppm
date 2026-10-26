// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++26 -triple %itanium_abi_triple -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++26 -triple %itanium_abi_triple -fmodule-file=a=%t/a.pcm -emit-llvm -o - %t/b.cpp | FileCheck %s

//--- a.cppm
export module a;

template<typename... Ts>
using element = Ts...[0];

export element<int> a = 0;

//--- b.cpp
import a;

int b() {
  return a;
}

// CHECK: @_ZW1a1a = external global i32
// CHECK: define {{.*}}i32 @_Z1bv()
// CHECK: load i32, ptr @_ZW1a1a
