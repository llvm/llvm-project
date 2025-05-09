// RUN: %clang_cc1 -std=c++11 -fptrauth-intrinsics -fptrauth-calls -emit-llvm -o - -triple=aarch64-windows-msvc %s | FileCheck %s

template <class T>
struct S {};

// CHECK: @"?s@@3U?$S@PE__ptrauth1A@ENC@AH@@A" =
S<int * __ptrauth(2, 0, 1234)> s;

// CHECK: define dso_local void @"?foo@@YAXPEAPE__ptrauth20OK@AH@Z"(
void foo(int * __ptrauth(3, 1, 234) *) {}

template <class T>
void foo(T t) {}

// CHECK: define weak_odr dso_local void @"??$foo@PEAPE__ptrauth0A@EA@AH@@YAXPEAPE__ptrauth0A@EA@AH@Z"(
template void foo<int * __ptrauth(1, 0, 64) *>(int * __ptrauth(1, 0, 64) *);

