// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm -o - %s | FileCheck %s

__declspec(dllimport) void f();
void g() { f(); } // use it

// CHECK: define dso_local dllexport void @f
void f() { }
