// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -triple %itanium_abi_triple %t/parta.cppm -o %t/mod-parta.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -triple %itanium_abi_triple %t/partb.cppm -o %t/mod-partb.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -triple %itanium_abi_triple %t/mod.cppm \
// RUN:   -fprebuilt-module-path=%t -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/mod.pcm -emit-llvm -disable-llvm-passes -o - \
// RUN:   -fprebuilt-module-path=%t | FileCheck %t/mod.cppm
// RUN: %clang_cc1 -std=c++20 -O2 -emit-module-interface -triple %itanium_abi_triple \
// RUN:   -fprebuilt-module-path=%t %t/mod.cppm -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -O2 -triple %itanium_abi_triple %t/mod.pcm -emit-llvm \
// RUN:   -fprebuilt-module-path=%t -disable-llvm-passes -o - | FileCheck %t/mod.cppm  -check-prefix=CHECK-OPT

//--- parta.cppm
export module mod:parta;

export int a = 43;

export int foo() {
  return 3 + a;
}

//--- partb.cppm
module mod:partb;

int b = 43;

int bar() {
  return 43 + b;
}

//--- mod.cppm
export module mod;
import :parta;
import :partb;
export int use() {
  return foo() + bar() + a + b;
}

// FIXME: The definition of the variables shouldn't be exported too.
// CHECK: @_ZW3mod1a = external global
// CHECK: @_ZW3mod1b = external global
// CHECK: declare{{.*}} i32 @_ZW3mod3foov
// CHECK: declare{{.*}} i32 @_ZW3mod3barv

// CHECK-OPT: @_ZW3mod1a = external global
// CHECK-OPT: @_ZW3mod1b = external global
// CHECK-OPT: declare{{.*}} i32 @_ZW3mod3foov
// CHECK-OPT: declare{{.*}} i32 @_ZW3mod3barv
