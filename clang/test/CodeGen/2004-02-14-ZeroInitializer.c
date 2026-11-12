// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s

// CHECK: zeroinitializer
int X[1000];
