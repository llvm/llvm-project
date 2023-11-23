// RUN: %clang_cc1 -triple aarch64 -target-feature +fp-armv8 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD
// RUN: %clang_cc1 -triple aarch64 -target-feature -fp-armv8 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SOFT

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

// No types are considered to be HFAs or HVAs by the soft-float PCS, so these
// are converted to integer types.
struct A {
  float x;
};
// SOFT: define dso_local i32 @test10(i64 %a.coerce)
// HARD: define dso_local %struct.A @test10([1 x float] alignstack(8) %a.coerce)
struct A test10(struct A a) { return a; }

struct B {
  double x;
  double y;
};
// SOFT: define dso_local [2 x i64] @test11([2 x i64] %a.coerce)
// HARD: define dso_local %struct.B @test11([2 x double] alignstack(8) %a.coerce)
struct B test11(struct B a) { return a; }

// Vector types are only available for targets with the correct hardware, and
// their calling-convention is left undefined by the soft-float ABI, so they
// aren't tested here.
