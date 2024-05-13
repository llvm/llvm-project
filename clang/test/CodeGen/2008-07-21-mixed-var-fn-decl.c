// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

int g0, f0(void);
int f1(void), g1;

// CHECK: @g0 = {{(dso_local )?}}global i32 0, align 4
// CHECK: @g1 = {{(dso_local )?}}global i32 0, align 4

