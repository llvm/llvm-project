// REQUIRES: !system-windows
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Empty.cppm \
// RUN:     -emit-module-interface -o %t/Empty.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/Empty2.cppm \
// RUN:     -fprebuilt-module-path=%t -emit-module-interface -o %t/Empty2.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/main.cpp \
// RUN:     -fprebuilt-module-path=%t -emit-llvm -o - | FileCheck %t/main.cpp
// RUN: %clang_cc1 -std=c++20  -triple %itanium_abi_triple %t/Empty2.pcm \
// RUN:     -fprebuilt-module-path=%t -emit-llvm -o - | FileCheck %t/Empty2.cppm

//--- Empty.cppm
export module Empty;

//--- Empty2.cppm
export module Empty2;
import Empty;

// CHECK-NOT: _ZGIW5Empty

//--- main.cpp
import Empty;
import Empty2;

// CHECK-NOT: _ZGIW5Empty
// CHECK-NOT: _ZGIW6Empty2
