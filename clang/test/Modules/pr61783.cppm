// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-windows-msvc19.11.0 -fms-extensions %t/mod.cppm -emit-module-interface \
// RUN:     -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-windows-msvc19.11.0 -fms-extensions %t/mod.pcm -emit-llvm -o - | \
// RUN:     FileCheck %t/mod.cppm
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-windows-msvc19.11.0 -fms-extensions %t/user.cpp -fmodule-file=mod=%t/mod.pcm \
// RUN:     -emit-llvm -o - | FileCheck %t/user.cpp

// Test again with reduced BMI
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-windows-msvc19.11.0 -fms-extensions %t/mod.cppm -emit-reduced-module-interface \
// RUN:     -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-windows-msvc19.11.0 -fms-extensions %t/mod.pcm -emit-llvm -o - | \
// RUN:     FileCheck %t/mod.cppm
// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-windows-msvc19.11.0 -fms-extensions %t/user.cpp -fmodule-file=mod=%t/mod.pcm \
// RUN:     -emit-llvm -o - | FileCheck %t/user.cpp

//--- mod.cppm
module;

#pragma comment(lib, "msvcprt.lib")
#pragma detect_mismatch("myLib_version", "9")

export module mod;

// CHECK: ![[NUM:[0-9]+]] ={{.*}}msvcprt.lib
// CHECK: ![[NUM:[0-9]+]] ={{.*}}FAILIFMISMATCH{{.*}}myLib_version=9

//--- user.cpp
#pragma detect_mismatch("myLib_version", "1")
import mod;

// CHECK: ![[NUM:[0-9]+]] ={{.*}}FAILIFMISMATCH{{.*}}myLib_version=1
// CHECK: ![[NUM:[0-9]+]] ={{.*}}msvcprt.lib
// CHECK: ![[NUM:[0-9]+]] ={{.*}}FAILIFMISMATCH{{.*}}myLib_version=9
