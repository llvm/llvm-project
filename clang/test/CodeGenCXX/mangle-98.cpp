// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -std=c++98 | FileCheck %s

template <bool B> struct S3 {};

// CHECK: define void @_Z1f2S3ILb1EE
void f(S3<true>) {}

// CHECK: define void @_Z1f2S3ILb0EE
void f(S3<false>) {}

// CHECK: define void @_Z2f22S3ILb1EE
void f2(S3<100>) {}
