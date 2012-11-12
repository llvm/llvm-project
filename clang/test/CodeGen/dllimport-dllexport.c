// RUN: %clang_cc1 -triple i386-mingw32 -emit-llvm < %s | FileCheck %s

void __attribute__((dllimport)) foo1();
void __attribute__((dllexport)) foo1(){}
// CHECK: define dllexport void @foo1
void __attribute__((dllexport)) foo2();

// PR6269
__declspec(dllimport) void foo3();
__declspec(dllexport) void foo3(){}
// CHECK: define dllexport void @foo3
__declspec(dllexport) void foo4();
