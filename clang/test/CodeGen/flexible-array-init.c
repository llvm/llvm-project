// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

struct { int x; int y[]; } a = { 1, 7, 11 };
// CHECK: @a ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 7, i32 11] }

struct { int x; int y[]; } b = { 1, { 13, 15 } };
// CHECK: @b ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 13, i32 15] }
