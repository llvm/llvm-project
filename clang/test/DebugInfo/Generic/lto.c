// RUN: %clang_cc1 -flto -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
// RUN: %clang_cc1 -flto=thin -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
// The "o" in LTO stands for optimization!
// CHECK: !DICompileUnit({{.*}} isOptimized: true
