// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

int g0, f0();
int f1(), g1;

// CHECK: @g0 = common {{(dso_local )?}}global i32 0, align 4
// CHECK: @g1 = common {{(dso_local )?}}global i32 0, align 4

