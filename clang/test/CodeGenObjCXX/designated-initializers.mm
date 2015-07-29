// RUN: %clang_cc1 -triple arm64 %s -verify -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// Make sure we don't enter an infinite loop (rdar://21942503)

int vals1[] = {
  [__objc_yes] = 1,
  [__objc_no] = 2
};
// CHECK: @vals1 = global [2 x i32] [i32 2, i32 1]

int vals2[] = {
  [true] = 3,
  [false] = 4
};
// CHECK: @vals2 = global [2 x i32] [i32 4, i32 3]

int vals3[] = {
  [false] = 1,
  [true] = 2,
  5
};
// CHECK: @vals3 = global [3 x i32] [i32 1, i32 2, i32 5]

int vals4[2] = {
  [true] = 5,
  [false] = 6
};
// CHECK: @vals4 = global [2 x i32] [i32 6, i32 5]

int vals5[3] = {
  [false] = 1,
  [true] = 2,
  6
};
// CHECK: @vals5 = global [3 x i32] [i32 1, i32 2, i32 6]
