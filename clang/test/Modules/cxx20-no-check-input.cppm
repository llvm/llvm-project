// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/use.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/a.pcm -emit-llvm -o - | FileCheck %t/a.ll
//
// RUN: echo "//Update" >> %t/foo.h
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/use.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/a.pcm -emit-llvm -o - | FileCheck %t/a.ll
//
// RUN: echo "//Update" >> %t/a.cppm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/use.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/a.pcm -emit-llvm -o - | FileCheck %t/a.ll
//
// RUN: rm -f %t/foo.h
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/use.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/a.pcm -emit-llvm -o - | FileCheck %t/a.ll
//
// RUN: rm -f %t/a.cppm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/use.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/a.pcm -emit-llvm -o - | FileCheck %t/a.ll

//--- foo.h
inline int foo = 43;

//--- a.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module a;
export using ::foo;

//--- a.ll
// check the LLVM IR are generated succesfully.
// CHECK: define{{.*}}@_ZGIW1a

//--- use.cpp
// expected-no-diagnostics
import a;
int use() {
    return foo;
}

