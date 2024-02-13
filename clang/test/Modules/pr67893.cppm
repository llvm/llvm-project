// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/a.cppm \
// RUN:      -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/m.cppm \
// RUN:      -emit-module-interface -fprebuilt-module-path=%t
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/m.pcm  \
// RUN:      -fprebuilt-module-path=%t -S -emit-llvm -o - | FileCheck %t/m.cppm

//--- a.cppm
export module a;
export struct A {
  A(){};
};
export A __dynamic_inited_a;

//--- m.cppm
module;
import a;
export module m;
import a;
module :private;
import a;

// CHECK: define void @_ZGIW1m
// CHECK: store i8 1, ptr @_ZGIW1m__in_chrg
// CHECK: call{{.*}}@_ZGIW1a
