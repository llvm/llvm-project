// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/a.cppm \
// RUN:      -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/m.cppm \
// RUN:      -emit-module-interface -fmodule-file=a=%t/a.pcm -o %t/m.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/m.pcm  \
// RUN:      -fmodule-file=a=%t/a.pcm -emit-llvm -o - | FileCheck %t/m.cppm

// Test again with reduced BMI.
// Note that we can't use reduced BMI here for m.cppm since it is required
// to generate the backend code.
// RUN: rm %t/a.pcm %t/m.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/a.cppm \
// RUN:      -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/m.cppm \
// RUN:      -emit-module-interface -fmodule-file=a=%t/a.pcm -o %t/m.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/m.pcm  \
// RUN:      -fmodule-file=a=%t/a.pcm -emit-llvm -o - | FileCheck %t/m.cppm

//--- a.cppm
export module a;
export struct A {
  A(){};
};
export A __dynamic_inited_a;

//--- m.cppm
export module m;
import a;
export import a;


// CHECK: define void @_ZGIW1m
// CHECK: store i8 1, ptr @_ZGIW1m__in_chrg
// CHECK: call{{.*}}@_ZGIW1a
