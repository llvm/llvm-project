// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK: @llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @foo to i8*), i8* bitcast (i32* @X to i8*)], section "llvm.metadata"
int X __attribute__((used));
int Y;

__attribute__((used)) void foo() {}
