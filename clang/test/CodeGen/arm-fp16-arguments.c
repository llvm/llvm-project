// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs -mfloat-abi soft -fallow-half-arguments-and-returns -emit-llvm -o - -O1 %s | FileCheck %s --check-prefix=CHECK --check-prefix=SOFT
// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs -mfloat-abi hard -fallow-half-arguments-and-returns -emit-llvm -o - -O1 %s | FileCheck %s --check-prefix=CHECK --check-prefix=HARD
// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs -mfloat-abi soft -fnative-half-arguments-and-returns -emit-llvm -o - -O1 %s | FileCheck %s --check-prefix=NATIVE

__fp16 g;

void t1(__fp16 a) { g = a; }
// SOFT: define void @t1(i32 [[PARAM:%.*]])
// SOFT: [[TRUNC:%.*]] = trunc i32 [[PARAM]] to i16
// HARD: define arm_aapcs_vfpcc void @t1(float [[PARAM:%.*]])
// HARD: [[BITCAST:%.*]] = bitcast float [[PARAM]] to i32
// HARD: [[TRUNC:%.*]] = trunc i32 [[BITCAST]] to i16
// CHECK: store i16 [[TRUNC]], i16* bitcast (half* @g to i16*)
// NATIVE: define void @t1(half [[PARAM:%.*]])
// NATIVE: store half [[PARAM]], half* @g

__fp16 t2() { return g; }
// SOFT: define i32 @t2()
// HARD: define arm_aapcs_vfpcc float @t2()
// NATIVE: define half @t2()
// CHECK: [[LOAD:%.*]] = load i16, i16* bitcast (half* @g to i16*)
// CHECK: [[ZEXT:%.*]] = zext i16 [[LOAD]] to i32
// SOFT: ret i32 [[ZEXT]]
// HARD: [[BITCAST:%.*]] = bitcast i32 [[ZEXT]] to float
// HARD: ret float [[BITCAST]]
// NATIVE: [[LOAD:%.*]] = load half, half* @g
// NATIVE: ret half [[LOAD]]
