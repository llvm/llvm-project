// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_Count
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple aarch64-windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_Count

// MSVC STL headers include only <intrin0.h>, which declares these four.

#include <intrin0.h>

unsigned int test_clz(unsigned long x) { return _CountLeadingZeros(x); }
// CHECK-LABEL: @test_clz(
// CHECK: call i32 @llvm.ctlz.i32(

unsigned int test_clz64(unsigned __int64 x) { return _CountLeadingZeros64(x); }
// CHECK-LABEL: @test_clz64(
// CHECK: call i64 @llvm.ctlz.i64(

unsigned int test_ctz(unsigned long x) { return _CountTrailingZeros(x); }
// CHECK-LABEL: @test_ctz(
// CHECK: call i32 @llvm.cttz.i32(

unsigned int test_ctz64(unsigned __int64 x) { return _CountTrailingZeros64(x); }
// CHECK-LABEL: @test_ctz64(
// CHECK: call i64 @llvm.cttz.i64(
