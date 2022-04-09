// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fblocks -emit-llvm -o - %s | FileCheck %s

struct S {
  S(const struct S &) {}
};

void (^b)(S) = ^(S) {};

// CHECK: [[DESCRIPTOR:%.*]] = getelementptr inbounds <{ i8*, %struct.S, [3 x i8] }>, <{ i8*, %struct.S, [3 x i8] }>* %0, i32 0, i32 0
// CHECK: load i8*, i8** [[DESCRIPTOR]]

