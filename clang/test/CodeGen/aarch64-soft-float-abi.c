// RUN: %clang_cc1 -triple aarch64 -target-feature +fp-armv8 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple aarch64 -target-feature -fp-armv8 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK

// Floats are passed in integer registers, this will be handled by the backend.
// CHECK: define dso_local half @test0(half noundef %a)
// CHECK: define dso_local bfloat @test1(bfloat noundef %a)
// CHECK: define dso_local float @test2(float noundef %a)
// CHECK: define dso_local double @test3(double noundef %a)
// CHECK: define dso_local fp128 @test4(fp128 noundef %a)
__fp16 test0(__fp16 a) { return a; }
__bf16 test1(__bf16 a) { return a; }
float test2(float a) { return a; }
double test3(double a) { return a; }
long double test4(long double a) { return a; }

