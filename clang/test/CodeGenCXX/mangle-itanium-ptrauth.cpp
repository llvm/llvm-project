// RUN: %clang_cc1 -std=c++11 -fptrauth-intrinsics -fptrauth-calls -emit-llvm -o - -triple=arm64-apple-ios %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -fptrauth-intrinsics -fptrauth-calls -emit-llvm -o - -triple=aarch64-linux-gnu %s | FileCheck %s

// CHECK: define {{.*}}void @_Z3fooPU9__ptrauthILj3ELb1ELj234EEPi(
void foo(int * __ptrauth(3, 1, 234) *) {}

template <class T>
void foo(T t) {}

// CHECK: define weak_odr void @_Z3fooIPU9__ptrauthILj1ELb0ELj64EEPiEvT_(
template void foo<int * __ptrauth(1, 0, 64) *>(int * __ptrauth(1, 0, 64) *);

