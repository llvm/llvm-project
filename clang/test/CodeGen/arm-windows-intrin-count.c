// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_Count

// The _Count* intrinsics, as on AArch64.

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

unsigned test_CountLeadingZeros(unsigned long x) { return _CountLeadingZeros(x); }
// CHECK-LABEL: @test_CountLeadingZeros(
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)

unsigned test_CountLeadingZeros64(unsigned __int64 x) { return _CountLeadingZeros64(x); }
// CHECK-LABEL: @test_CountLeadingZeros64(
// CHECK: [[R:%.*]] = call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// CHECK: trunc i64 [[R]] to i32

unsigned test_CountLeadingOnes(unsigned long x) { return _CountLeadingOnes(x); }
// CHECK-LABEL: @test_CountLeadingOnes(
// CHECK: [[N:%.*]] = xor i32 %{{.*}}, -1
// CHECK: call i32 @llvm.ctlz.i32(i32 [[N]], i1 false)

unsigned test_CountLeadingOnes64(unsigned __int64 x) { return _CountLeadingOnes64(x); }
// CHECK-LABEL: @test_CountLeadingOnes64(
// CHECK: [[N:%.*]] = xor i64 %{{.*}}, -1
// CHECK: [[R:%.*]] = call i64 @llvm.ctlz.i64(i64 [[N]], i1 false)
// CHECK: trunc i64 [[R]] to i32

unsigned test_CountLeadingSigns(long x) { return _CountLeadingSigns(x); }
// CHECK-LABEL: @test_CountLeadingSigns(
// CHECK: call i32 @llvm.arm.cls(i32 %{{.*}})

unsigned test_CountLeadingSigns64(__int64 x) { return _CountLeadingSigns64(x); }
// CHECK-LABEL: @test_CountLeadingSigns64(
// CHECK: call i32 @llvm.arm.cls64(i64 %{{.*}})

unsigned test_CountOneBits(unsigned long x) { return _CountOneBits(x); }
// CHECK-LABEL: @test_CountOneBits(
// CHECK: call i32 @llvm.ctpop.i32(i32 %{{.*}})

unsigned test_CountOneBits64(unsigned __int64 x) { return _CountOneBits64(x); }
// CHECK-LABEL: @test_CountOneBits64(
// CHECK: [[R:%.*]] = call i64 @llvm.ctpop.i64(i64 %{{.*}})
// CHECK: trunc i64 [[R]] to i32

unsigned test_CountTrailingZeros(unsigned long x) { return _CountTrailingZeros(x); }
// CHECK-LABEL: @test_CountTrailingZeros(
// CHECK: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)

unsigned test_CountTrailingZeros64(unsigned __int64 x) { return _CountTrailingZeros64(x); }
// CHECK-LABEL: @test_CountTrailingZeros64(
// CHECK: [[R:%.*]] = call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// CHECK: trunc i64 [[R]] to i32
