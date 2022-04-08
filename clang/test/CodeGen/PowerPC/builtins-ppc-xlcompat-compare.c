// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefix=CHECK-64B
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s --check-prefix=CHECK-64B
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefix=CHECK-32B
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefix=CHECK-64B

// CHECK-64B-LABEL: @test_builtin_ppc_cmpb(
// CHECK-64B:         [[LLA_ADDR:%.*]] = alloca i64, align 8
// CHECK-64B-NEXT:    [[LLB_ADDR:%.*]] = alloca i64, align 8
// CHECK-64B-NEXT:    store i64 [[LLA:%.*]], i64* [[LLA_ADDR]], align 8
// CHECK-64B-NEXT:    store i64 [[LLB:%.*]], i64* [[LLB_ADDR]], align 8
// CHECK-64B-NEXT:    [[TMP0:%.*]] = load i64, i64* [[LLA_ADDR]], align 8
// CHECK-64B-NEXT:    [[TMP1:%.*]] = load i64, i64* [[LLB_ADDR]], align 8
// CHECK-64B-NEXT:    [[CMPB:%.*]] = call i64 @llvm.ppc.cmpb.i64.i64.i64(i64 [[TMP0]], i64 [[TMP1]])
// CHECK-64B-NEXT:    ret i64 [[CMPB]]
//
// CHECK-32B-LABEL: @test_builtin_ppc_cmpb(
// CHECK-32B:         [[LLA_ADDR:%.*]] = alloca i64, align 8
// CHECK-32B-NEXT:    [[LLB_ADDR:%.*]] = alloca i64, align 8
// CHECK-32B-NEXT:    store i64 [[LLA:%.*]], i64* [[LLA_ADDR]], align 8
// CHECK-32B-NEXT:    store i64 [[LLB:%.*]], i64* [[LLB_ADDR]], align 8
// CHECK-32B-NEXT:    [[TMP0:%.*]] = load i64, i64* [[LLA_ADDR]], align 8
// CHECK-32B-NEXT:    [[TMP1:%.*]] = load i64, i64* [[LLB_ADDR]], align 8
// CHECK-32B-NEXT:    [[TMP2:%.*]] = trunc i64 [[TMP0]] to i32
// CHECK-32B-NEXT:    [[TMP3:%.*]] = trunc i64 [[TMP1]] to i32
// CHECK-32B-NEXT:    [[TMP4:%.*]] = lshr i64 [[TMP0]], 32
// CHECK-32B-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
// CHECK-32B-NEXT:    [[TMP6:%.*]] = lshr i64 [[TMP1]], 32
// CHECK-32B-NEXT:    [[TMP7:%.*]] = trunc i64 [[TMP6]] to i32
// CHECK-32B-NEXT:    [[CMPB:%.*]] = call i32 @llvm.ppc.cmpb.i32.i32.i32(i32 [[TMP2]], i32 [[TMP3]])
// CHECK-32B-NEXT:    [[TMP8:%.*]] = zext i32 [[CMPB]] to i64
// CHECK-32B-NEXT:    [[CMPB1:%.*]] = call i32 @llvm.ppc.cmpb.i32.i32.i32(i32 [[TMP5]], i32 [[TMP7]])
// CHECK-32B-NEXT:    [[TMP9:%.*]] = zext i32 [[CMPB1]] to i64
// CHECK-32B-NEXT:    [[TMP10:%.*]] = shl i64 [[TMP9]], 32
// CHECK-32B-NEXT:    [[TMP11:%.*]] = or i64 [[TMP8]], [[TMP10]]
// CHECK-32B-NEXT:    ret i64 [[TMP11]]
//
long long test_builtin_ppc_cmpb(long long lla, long long llb) {
  return __builtin_ppc_cmpb(lla, llb);
}
