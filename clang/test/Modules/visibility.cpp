// Test that modules with different visibility mode can be shared.
// REQUIRES: aarch64-registered-target

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=default -fmodules -emit-module -fmodule-name=foo %t/foo.modulemap -o %t/foo.default.pcm
// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=hidden  -fmodules -emit-module -fmodule-name=foo %t/foo.modulemap -o %t/foo.hidden.pcm

// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=default -fmodules -fmodule-file=%t/foo.default.pcm -I%t -emit-llvm %t/test.cpp -o - | FileCheck %s --check-prefixes=DEFAULT,BOTH
// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=default -fmodules -fmodule-file=%t/foo.hidden.pcm  -I%t -emit-llvm %t/test.cpp -o - | FileCheck %s --check-prefixes=DEFAULT,BOTH

// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=hidden -fmodules -fmodule-file=%t/foo.default.pcm -I%t -emit-llvm %t/test.cpp -o - | FileCheck %s --check-prefixes=HIDDEN,BOTH
// RUN: %clang_cc1 -triple arm64e-apple-macos -x c++ -fvisibility=hidden -fmodules -fmodule-file=%t/foo.hidden.pcm  -I%t -emit-llvm %t/test.cpp -o - | FileCheck %s --check-prefixes=HIDDEN,BOTH

// DEFAULT: define void @_Z2f4v()
// HIDDEN: define hidden void @_Z2f4v()
// DEFAULT: define void @_Z4testv()
// HIDDEN: define hidden void @_Z4testv()
// BOTH: declare void @_Z2f1v()
// BOTH: define internal void @_ZL2f2v()
// DEFAULT: define linkonce_odr void @_Z2f3v()
// HIDDEN: define linkonce_odr hidden void @_Z2f3v()


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

