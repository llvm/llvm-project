// Test that modules with different visibility mode can be shared.
// REQUIRES: aarch64-registered-target

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=default -fmodules-codegen -fmodules -emit-module -fmodule-name=foo %t/foo.modulemap -o %t/foo.pcm

// RUN: %clang_cc1 -emit-llvm %t/foo.pcm -o - | FileCheck %s --check-prefix=DEF
// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=hidden -fmodules -fmodule-file=%t/foo.pcm -I%t -emit-llvm %t/test.cpp -o - | FileCheck %s --check-prefixes=USE

// DEF: define void @_Z2f4v()
// DEF: define weak_odr void @_Z2f3v()

// USE: define hidden void @_Z4testv()
// USE: declare void @_Z2f1v()
// USE: define internal void @_ZL2f2v()
// USE: declare void @_Z2f3v()
// USE: declare void @_Z2f4v()


//--- foo.h
void f1();
static void f2() {}
inline void f3() {}
void f4() {}

//--- test.cpp
#include "foo.h"

void test() {
  f1();
  f2();
  f3();
  f4();
}

//--- foo.modulemap
module foo { header "foo.h" }

