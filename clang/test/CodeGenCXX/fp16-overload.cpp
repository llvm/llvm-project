// RUN: %clang_cc1 -emit-llvm -o - -triple arm-none-linux-gnueabi %s | FileCheck %s

extern int foo(float x);
extern int foo(double x);

__fp16 a;

// CHECK: call noundef i32 @_Z3foof
// CHECK-NOT: call noundef i32 @_Z3food
int bar (void) { return foo(a); }
