// RUN: %clang_cc1 -triple aarch64 -target-feature +fp-armv8 -target-abi aapcs -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD
// RUN: %clang_cc1 -triple aarch64 -target-feature -fp-armv8 -target-abi aapcs-soft -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SOFT

// See also llvm/test/CodeGen/AArch64/soft-float-abi.ll, which checks the LLVM
// backend parts of the soft-float ABI.

// The va_list type does not change between the ABIs
// CHECK: %struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

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

#include <stdarg.h>

// The layout of the va_list struct is unchanged between the ABIs, but for
// aapcs-soft, floating-point arguments will be retreived from the GPR save
// area, as if they were an integer type of the same size.
// CHECK-LABEL: define dso_local double @test20(i32 noundef %a, ...)
// CHECK: %vl = alloca %struct.__va_list, align 8
// SOFT: %gr_offs_p = getelementptr inbounds nuw %struct.__va_list, ptr %vl, i32 0, i32 3
// SOFT: %reg_top_p = getelementptr inbounds nuw %struct.__va_list, ptr %vl, i32 0, i32 1
// HARD: %vr_offs_p = getelementptr inbounds nuw %struct.__va_list, ptr %vl, i32 0, i32 4
// HARD: %reg_top_p = getelementptr inbounds nuw %struct.__va_list, ptr %vl, i32 0, i32 2
double test20(int a, ...) {
  va_list vl;
  va_start(vl, a);
  return va_arg(vl, double);
}

// Vector types are only available for targets with the correct hardware, and
// their calling-convention is left undefined by the soft-float ABI, so they
// aren't tested here.
