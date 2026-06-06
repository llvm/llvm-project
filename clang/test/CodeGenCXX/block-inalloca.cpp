// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fblocks -emit-llvm -o - %s | FileCheck %s

struct S {
  S(const struct S &) {}
};

void (^b)(S) = ^(S) {};

// CHECK: [[DESCRIPTOR:%.*]] = getelementptr inbounds nuw <{ ptr, %struct.S, [3 x i8] }>, ptr %0, i32 0, i32 0
// CHECK: load ptr, ptr [[DESCRIPTOR]]

