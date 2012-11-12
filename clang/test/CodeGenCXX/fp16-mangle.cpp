// RUN: %clang_cc1 -emit-llvm -o - -triple arm-none-linux-gnueabi %s | FileCheck %s

// CHECK: @_ZN1SIDhDhE1iE = global i32 3
template <typename T, typename U> struct S { static int i; };
template <> int S<__fp16, __fp16>::i = 3;

// CHECK: define void @_Z1fPDh(i16* %x)
void f (__fp16 *x) { }

// CHECK: define void @_Z1gPDhS_(i16* %x, i16* %y)
void g (__fp16 *x, __fp16 *y) { }

