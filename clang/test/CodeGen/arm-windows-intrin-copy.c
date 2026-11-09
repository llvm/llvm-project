// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_Copy

// The _Copy* intrinsics move a bit pattern between integer and float.

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

double test_CopyDoubleFromInt64(__int64 x) { return _CopyDoubleFromInt64(x); }
// CHECK-LABEL: @test_CopyDoubleFromInt64(
// CHECK: [[R:%.*]] = bitcast i64 %{{.*}} to double
// CHECK: ret double [[R]]

float test_CopyFloatFromInt32(__int32 x) { return _CopyFloatFromInt32(x); }
// CHECK-LABEL: @test_CopyFloatFromInt32(
// CHECK: [[R:%.*]] = bitcast i32 %{{.*}} to float
// CHECK: ret float [[R]]

__int32 test_CopyInt32FromFloat(float x) { return _CopyInt32FromFloat(x); }
// CHECK-LABEL: @test_CopyInt32FromFloat(
// CHECK: [[R:%.*]] = bitcast float %{{.*}} to i32
// CHECK: ret i32 [[R]]

__int64 test_CopyInt64FromDouble(double x) { return _CopyInt64FromDouble(x); }
// CHECK-LABEL: @test_CopyInt64FromDouble(
// CHECK: [[R:%.*]] = bitcast double %{{.*}} to i64
// CHECK: ret i64 [[R]]
