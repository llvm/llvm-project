// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefix=NO_HALF


float abs_float(float X) {
  return abs(X);
}

// CHECK: define noundef float @"?abs_float@@YAMM@Z"(
// CHECK: call float @llvm.fabs.f32(float %0)

double abs_double(double X) {
  return abs(X);
}

// CHECK: define noundef double @"?abs_double@@YANN@Z"(
// CHECK: call double @llvm.fabs.f64(double %0)

half abs_half(half X) {
  return abs(X);
}

// CHECK: define noundef half @"?abs_half@@YA$f16@$f16@@Z"(
// CHECK: call half @llvm.fabs.f16(half %0)
// NO_HALF: define noundef float @"?abs_half@@YA$halff@$halff@@Z"(
// NO_HALF: call float @llvm.fabs.f32(float %0)

int abs_int(int X) {
  return abs(X);
}

// NO_HALF: define noundef i32 @"?abs_int@@YAHH@Z"(i32
// CHECK: define noundef i32 @"?abs_int@@YAHH@Z"(i32
// CHECK:         [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[A:%.*]], ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT:    [[NEG:%.*]] = sub nsw i32 0, [[TMP0]]
// CHECK-NEXT:    [[ABSCOND:%.*]] = icmp slt i32 [[TMP0]], 0
// CHECK-NEXT:    [[ABS:%.*]] = select i1 [[ABSCOND]], i32 [[NEG]], i32 [[TMP0]]
// CHECK-NEXT:    ret i32 [[ABS]]

int64_t abs_int64(int64_t X) {
  return abs(X);
}

// CHECK: define noundef i64 @"?abs_int64@@YAJJ@Z"(i64
// CHECK:         [[A_ADDR:%.*]] = alloca i64, align 8
// CHECK-NEXT:    store i64 [[A:%.*]], ptr [[A_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i64, ptr [[A_ADDR]], align 8
// CHECK-NEXT:    [[NEG:%.*]] = sub nsw i64 0, [[TMP0]]
// CHECK-NEXT:    [[ABSCOND:%.*]] = icmp slt i64 [[TMP0]], 0
// CHECK-NEXT:    [[ABS:%.*]] = select i1 [[ABSCOND]], i64 [[NEG]], i64 [[TMP0]]
// CHECK-NEXT:    ret i64 [[ABS]]
