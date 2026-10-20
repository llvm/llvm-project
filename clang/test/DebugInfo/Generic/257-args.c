// RUN: %clang_cc1 -x c++ -debug-info-kind=limited -emit-llvm -triple x86_64-linux-gnu -o - %s | FileCheck %s
// PR23332

// CHECK: DILocalVariable(arg: 255
// CHECK: DILocalVariable(arg: 256
// CHECK: DILocalVariable(arg: 257
void fn1(int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int) {}
