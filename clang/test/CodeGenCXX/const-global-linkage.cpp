// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

const int x = 10;
const int y = 20;
const volatile int z = 30;
// CHECK-NOT: @x
// CHECK: @z = constant i32 30
// CHECK: @_ZL1y = internal constant i32 20
const int& b() { return y; }

const char z1[] = "asdf";
const char z2[] = "zxcv";
const volatile char z3[] = "zxcv";
// CHECK-NOT: @z1
// CHECK: @z3 = constant
// CHECK: @_ZL2z2 = internal constant
const char* b2() { return z2; }
