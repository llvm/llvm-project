// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-w64-mingw32 | FileCheck %s

int func() { return 0; }
// CHECK-DAG: @_Z4funcv

int main() { return 0; }
// CHECK-DAG: @main

int wmain() { return 0; }
// CHECK-DAG: @wmain

int WinMain() { return 0; }
// CHECK-DAG: @WinMain

int wWinMain() { return 0; }
// CHECK-DAG: @wWinMain

int DllMain() { return 0; }
// CHECK-DAG: @DllMain
