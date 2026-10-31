// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -fmodules \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/main.cpp 2>&1 | FileCheck %s

// CHECK: main.cpp:1:15: error: redefinition of module 'M'
// CHECK: module.modulemap:1:8: note: previously defined here
// CHECK: 1 error generated.

//--- module.modulemap
module M {}

//--- main.cpp
export module M;
