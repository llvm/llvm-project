// RUN: %clang_cc1 -emit-llvm -w -x objective-c %s -o - | FileCheck %s
// rdar://7095855 rdar://7115749

// CHECK: internal unnamed_addr constant [6 x i16] [i16 105, i16 80, i16 111, i16 100, i16 8482, i16 0], align 2
void *P = @"iPod™";
