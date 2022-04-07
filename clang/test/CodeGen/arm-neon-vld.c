// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:     -S -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | \
// RUN:     FileCheck -check-prefixes=CHECK,CHECK-A64 %s
// RUN: %clang_cc1 -no-opaque-pointers -triple armv8-none-linux-gnueabi -target-feature +neon \
// RUN:     -target-feature +fp16 -S -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:     opt -S -mem2reg | FileCheck -check-prefixes=CHECK,CHECK-A32 %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: @test_vld1_f16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK-A32: %struct.float16x4x2_t* noalias sret(%struct.float16x4x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to [[HALF:(half|i16)]]*
// CHECK: [[VLD1XN:%.*]] = call { <4 x [[HALF]]>, <4 x [[HALF]]> } @llvm.{{aarch64.neon.ld1x2.v4f16.p0f16|arm.neon.vld1x2.v4i16.p0i16}}([[HALF]]* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x [[HALF]]>, <4 x [[HALF]]> }*
// CHECK: store { <4 x [[HALF]]>, <4 x [[HALF]]> } [[VLD1XN]], { <4 x [[HALF]]>, <4 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x4x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float16x4x2_t, %struct.float16x4x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.float16x4x2_t [[TMP6]]
// CHECK-A32: ret void
float16x4x2_t test_vld1_f16_x2(float16_t const *a) {
  return vld1_f16_x2(a);
}

// CHECK-LABEL: @test_vld1_f16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK-A32: %struct.float16x4x3_t* noalias sret(%struct.float16x4x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to [[HALF]]*
// CHECK: [[VLD1XN:%.*]] = call { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> } @llvm.{{aarch64.neon.ld1x3.v4f16.p0f16|arm.neon.vld1x3.v4i16.p0i16}}([[HALF]]* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }*
// CHECK: store { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> } [[VLD1XN]], { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x4x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float16x4x3_t, %struct.float16x4x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.float16x4x3_t [[TMP6]]
// CHECK-A32: ret void
float16x4x3_t test_vld1_f16_x3(float16_t const *a) {
  return vld1_f16_x3(a);
}

// CHECK-LABEL: @test_vld1_f16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK-A32: %struct.float16x4x4_t* noalias sret(%struct.float16x4x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to [[HALF]]*
// CHECK: [[VLD1XN:%.*]] = call { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> } @llvm.{{aarch64.neon.ld1x4.v4f16.p0f16|arm.neon.vld1x4.v4i16.p0i16}}([[HALF]]* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }*
// CHECK: store { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> } [[VLD1XN]], { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x4x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float16x4x4_t, %struct.float16x4x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.float16x4x4_t [[TMP6]]
// CHECK-A32: ret void
float16x4x4_t test_vld1_f16_x4(float16_t const *a) {
  return vld1_f16_x4(a);
}

// CHECK-LABEL: @test_vld1_f32_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK-A32: %struct.float32x2x2_t* noalias sret(%struct.float32x2x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK: [[VLD1XN:%.*]] = call { <2 x float>, <2 x float> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v2f32.p0f32(float* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x float>, <2 x float> }*
// CHECK: store { <2 x float>, <2 x float> } [[VLD1XN]], { <2 x float>, <2 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x2x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.float32x2x2_t [[TMP6]]
// CHECK-A32: ret void
float32x2x2_t test_vld1_f32_x2(float32_t const *a) {
  return vld1_f32_x2(a);
}

// CHECK-LABEL: @test_vld1_f32_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK-A32: %struct.float32x2x3_t* noalias sret(%struct.float32x2x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK: [[VLD1XN:%.*]] = call { <2 x float>, <2 x float>, <2 x float> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v2f32.p0f32(float* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x float>, <2 x float>, <2 x float> }*
// CHECK: store { <2 x float>, <2 x float>, <2 x float> } [[VLD1XN]], { <2 x float>, <2 x float>, <2 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x2x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float32x2x3_t, %struct.float32x2x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.float32x2x3_t [[TMP6]]
float32x2x3_t test_vld1_f32_x3(float32_t const *a) {
  return vld1_f32_x3(a);
}

// CHECK-LABEL: @test_vld1_f32_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK-A32: %struct.float32x2x4_t* noalias sret(%struct.float32x2x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK: [[VLD1XN:%.*]] = call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v2f32.p0f32(float* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x float>, <2 x float>, <2 x float>, <2 x float> }*
// CHECK: store { <2 x float>, <2 x float>, <2 x float>, <2 x float> } [[VLD1XN]], { <2 x float>, <2 x float>, <2 x float>, <2 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x2x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float32x2x4_t, %struct.float32x2x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.float32x2x4_t [[TMP6]]
// CHECK-A32: ret void
float32x2x4_t test_vld1_f32_x4(float32_t const *a) {
  return vld1_f32_x4(a);
}

// CHECK-LABEL: @test_vld1_p16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK-A32: %struct.poly16x4x2_t* noalias sret(%struct.poly16x4x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x4x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.poly16x4x2_t [[TMP6]]
// CHECK-A32: ret void
poly16x4x2_t test_vld1_p16_x2(poly16_t const *a) {
  return vld1_p16_x2(a);
}

// CHECK-LABEL: @test_vld1_p16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK-A32: %struct.poly16x4x3_t* noalias sret(%struct.poly16x4x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x4x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.poly16x4x3_t [[TMP6]]
// CHECK-A32: ret void
poly16x4x3_t test_vld1_p16_x3(poly16_t const *a) {
  return vld1_p16_x3(a);
}

// CHECK-LABEL: @test_vld1_p16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK-A32: %struct.poly16x4x4_t* noalias sret(%struct.poly16x4x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x4x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.poly16x4x4_t [[TMP6]]
// CHECK-A32: ret void
poly16x4x4_t test_vld1_p16_x4(poly16_t const *a) {
  return vld1_p16_x4(a);
}

// CHECK-LABEL: @test_vld1_p8_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK-A32: %struct.poly8x8x2_t* noalias sret(%struct.poly8x8x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x8x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.poly8x8x2_t [[TMP4]]
// CHECK-A32: ret void
poly8x8x2_t test_vld1_p8_x2(poly8_t const *a) {
  return vld1_p8_x2(a);
}

// CHECK-LABEL: @test_vld1_p8_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK-A32: %struct.poly8x8x3_t* noalias sret(%struct.poly8x8x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x8x3_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x8x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.poly8x8x3_t [[TMP4]]
// CHECK-A32: ret void
poly8x8x3_t test_vld1_p8_x3(poly8_t const *a) {
  return vld1_p8_x3(a);
}

// CHECK-LABEL: @test_vld1_p8_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK-A32: %struct.poly8x8x4_t* noalias sret(%struct.poly8x8x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x8x4_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x8x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.poly8x8x4_t [[TMP4]]
// CHECK-A32: ret void
poly8x8x4_t test_vld1_p8_x4(poly8_t const *a) {
  return vld1_p8_x4(a);
}

// CHECK-LABEL: @test_vld1_s16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK-A32: %struct.int16x4x2_t* noalias sret(%struct.int16x4x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x4x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int16x4x2_t [[TMP6]]
// CHECK-A32: ret void
int16x4x2_t test_vld1_s16_x2(int16_t const *a) {
  return vld1_s16_x2(a);
}

// CHECK-LABEL: @test_vld1_s16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK-A32: %struct.int16x4x3_t* noalias sret(%struct.int16x4x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x4x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int16x4x3_t, %struct.int16x4x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int16x4x3_t [[TMP6]]
// CHECK-A32: ret void
int16x4x3_t test_vld1_s16_x3(int16_t const *a) {
  return vld1_s16_x3(a);
}

// CHECK-LABEL: @test_vld1_s16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK-A32: %struct.int16x4x4_t* noalias sret(%struct.int16x4x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x4x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int16x4x4_t, %struct.int16x4x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int16x4x4_t [[TMP6]]
// CHECK-A32: ret void
int16x4x4_t test_vld1_s16_x4(int16_t const *a) {
  return vld1_s16_x4(a);
}

// CHECK-LABEL: @test_vld1_s32_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK-A32: %struct.int32x2x2_t* noalias sret(%struct.int32x2x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v2i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32> } [[VLD1XN]], { <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x2x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int32x2x2_t [[TMP6]]
// CHECK-A32: ret void
int32x2x2_t test_vld1_s32_x2(int32_t const *a) {
  return vld1_s32_x2(a);
}

// CHECK-LABEL: @test_vld1_s32_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK-A32: %struct.int32x2x3_t* noalias sret(%struct.int32x2x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v2i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD1XN]], { <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x2x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int32x2x3_t, %struct.int32x2x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int32x2x3_t [[TMP6]]
// CHECK-A32: ret void
int32x2x3_t test_vld1_s32_x3(int32_t const *a) {
  return vld1_s32_x3(a);
}

// CHECK-LABEL: @test_vld1_s32_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK-A32: %struct.int32x2x4_t* noalias sret(%struct.int32x2x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v2i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD1XN]], { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x2x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int32x2x4_t, %struct.int32x2x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int32x2x4_t [[TMP6]]
// CHECK-A32: ret void
int32x2x4_t test_vld1_s32_x4(int32_t const *a) {
  return vld1_s32_x4(a);
}

// CHECK-LABEL: @test_vld1_s64_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK-A32: %struct.int64x1x2_t* noalias sret(%struct.int64x1x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x1x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v1i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64> } [[VLD1XN]], { <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x1x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x1x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int64x1x2_t, %struct.int64x1x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int64x1x2_t [[TMP6]]
// CHECK-A32: ret void
int64x1x2_t test_vld1_s64_x2(int64_t const *a) {
  return vld1_s64_x2(a);
}

// CHECK-LABEL: @test_vld1_s64_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK-A32: %struct.int64x1x3_t* noalias sret(%struct.int64x1x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x1x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v1i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x1x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x1x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int64x1x3_t, %struct.int64x1x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int64x1x3_t [[TMP6]]
// CHECK-A32: ret void
int64x1x3_t test_vld1_s64_x3(int64_t const *a) {
  return vld1_s64_x3(a);
}

// CHECK-LABEL: @test_vld1_s64_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK-A32: %struct.int64x1x4_t* noalias sret(%struct.int64x1x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x1x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v1i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x1x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x1x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int64x1x4_t, %struct.int64x1x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int64x1x4_t [[TMP6]]
// CHECK-A32: ret void
int64x1x4_t test_vld1_s64_x4(int64_t const *a) {
  return vld1_s64_x4(a);
}

// CHECK-LABEL: @test_vld1_s8_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK-A32: %struct.int8x8x2_t* noalias sret(%struct.int8x8x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x8x2_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x8x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int8x8x2_t [[TMP4]]
// CHECK-A32: ret void
int8x8x2_t test_vld1_s8_x2(int8_t const *a) {
  return vld1_s8_x2(a);
}

// CHECK-LABEL: @test_vld1_s8_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK-A32: %struct.int8x8x3_t* noalias sret(%struct.int8x8x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x8x3_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x8x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.int8x8x3_t, %struct.int8x8x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int8x8x3_t [[TMP4]]
// CHECK-A32: ret void
int8x8x3_t test_vld1_s8_x3(int8_t const *a) {
  return vld1_s8_x3(a);
}

// CHECK-LABEL: @test_vld1_s8_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK-A32: %struct.int8x8x4_t* noalias sret(%struct.int8x8x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x8x4_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x8x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.int8x8x4_t, %struct.int8x8x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.int8x8x4_t [[TMP4]]
// CHECK-A32: ret void
int8x8x4_t test_vld1_s8_x4(int8_t const *a) {
  return vld1_s8_x4(a);
}

// CHECK-LABEL: @test_vld1_u16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK-A32: %struct.uint16x4x2_t* noalias sret(%struct.uint16x4x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x4x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint16x4x2_t [[TMP6]]
// CHECK-A32: ret void
uint16x4x2_t test_vld1_u16_x2(uint16_t const *a) {
  return vld1_u16_x2(a);
}

// CHECK-LABEL: @test_vld1_u16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK-A32: %struct.uint16x4x3_t* noalias sret(%struct.uint16x4x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x4x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint16x4x3_t [[TMP6]]
// CHECK-A32: ret void
uint16x4x3_t test_vld1_u16_x3(uint16_t const *a) {
  return vld1_u16_x3(a);
}

// CHECK-LABEL: @test_vld1_u16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK-A32: %struct.uint16x4x4_t* noalias sret(%struct.uint16x4x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v4i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD1XN]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x4x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint16x4x4_t [[TMP6]]
// CHECK-A32: ret void
uint16x4x4_t test_vld1_u16_x4(uint16_t const *a) {
  return vld1_u16_x4(a);
}

// CHECK-LABEL: @test_vld1_u32_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK-A32: %struct.uint32x2x2_t* noalias sret(%struct.uint32x2x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v2i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32> } [[VLD1XN]], { <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x2x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint32x2x2_t [[TMP6]]
// CHECK-A32: ret void
uint32x2x2_t test_vld1_u32_x2(uint32_t const *a) {
  return vld1_u32_x2(a);
}

// CHECK-LABEL: @test_vld1_u32_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK-A32: %struct.uint32x2x3_t* noalias sret(%struct.uint32x2x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v2i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD1XN]], { <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x2x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint32x2x3_t [[TMP6]]
// CHECK-A32: ret void
uint32x2x3_t test_vld1_u32_x3(uint32_t const *a) {
  return vld1_u32_x3(a);
}

// CHECK-LABEL: @test_vld1_u32_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK-A32: %struct.uint32x2x4_t* noalias sret(%struct.uint32x2x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v2i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD1XN]], { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x2x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint32x2x4_t [[TMP6]]
// CHECK-A32: ret void
uint32x2x4_t test_vld1_u32_x4(uint32_t const *a) {
  return vld1_u32_x4(a);
}

// CHECK-LABEL: @test_vld1_u64_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK-A32: %struct.uint64x1x2_t* noalias sret(%struct.uint64x1x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x1x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v1i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64> } [[VLD1XN]], { <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x1x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x1x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint64x1x2_t [[TMP6]]
// CHECK-A32: ret void
uint64x1x2_t test_vld1_u64_x2(uint64_t const *a) {
  return vld1_u64_x2(a);
}

// CHECK-LABEL: @test_vld1_u64_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK-A32: %struct.uint64x1x3_t* noalias sret(%struct.uint64x1x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x1x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v1i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x1x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x1x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint64x1x3_t [[TMP6]]
// CHECK-A32: ret void
uint64x1x3_t test_vld1_u64_x3(uint64_t const *a) {
  return vld1_u64_x3(a);
}

// CHECK-LABEL: @test_vld1_u64_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK-A32: %struct.uint64x1x4_t* noalias sret(%struct.uint64x1x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x1x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v1i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x1x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x1x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint64x1x4_t [[TMP6]]
// CHECK-A32: ret void
uint64x1x4_t test_vld1_u64_x4(uint64_t const *a) {
  return vld1_u64_x4(a);
}

// CHECK-LABEL: @test_vld1_u8_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK-A32: %struct.uint8x8x2_t* noalias sret(%struct.uint8x8x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x8x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 16, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint8x8x2_t [[TMP4]]
// CHECK-A32: ret void
uint8x8x2_t test_vld1_u8_x2(uint8_t const *a) {
  return vld1_u8_x2(a);
}

// CHECK-LABEL: @test_vld1_u8_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK-A32: %struct.uint8x8x3_t* noalias sret(%struct.uint8x8x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x8x3_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x8x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 24, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint8x8x3_t [[TMP4]]
// CHECK-A32: ret void
uint8x8x3_t test_vld1_u8_x3(uint8_t const *a) {
  return vld1_u8_x3(a);
}

// CHECK-LABEL: @test_vld1_u8_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK-A32: %struct.uint8x8x4_t* noalias sret(%struct.uint8x8x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x8x4_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v8i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD1XN]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x8x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[RETVAL]], align 8
// CHECK-A64: ret %struct.uint8x8x4_t [[TMP4]]
// CHECK-A32: ret void
uint8x8x4_t test_vld1_u8_x4(uint8_t const *a) {
  return vld1_u8_x4(a);
}

// CHECK-LABEL: @test_vld1q_f16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK-A32: %struct.float16x8x2_t* noalias sret(%struct.float16x8x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to [[HALF]]*
// CHECK: [[VLD1XN:%.*]] = call { <8 x [[HALF]]>, <8 x [[HALF]]> } @llvm.{{aarch64.neon.ld1x2.v8f16.p0f16|arm.neon.vld1x2.v8i16.p0i16}}([[HALF]]* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x [[HALF]]>, <8 x [[HALF]]> }*
// CHECK: store { <8 x [[HALF]]>, <8 x [[HALF]]> } [[VLD1XN]], { <8 x [[HALF]]>, <8 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x8x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float16x8x2_t, %struct.float16x8x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.float16x8x2_t [[TMP6]]
// CHECK-A32: ret void
float16x8x2_t test_vld1q_f16_x2(float16_t const *a) {
  return vld1q_f16_x2(a);
}

// CHECK-LABEL: @test_vld1q_f16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK-A32: %struct.float16x8x3_t* noalias sret(%struct.float16x8x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to [[HALF]]*
// CHECK: [[VLD1XN:%.*]] = call { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> } @llvm.{{aarch64.neon.ld1x3.v8f16.p0f16|arm.neon.vld1x3.v8i16.p0i16}}([[HALF]]* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }*
// CHECK: store { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> } [[VLD1XN]], { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x8x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float16x8x3_t, %struct.float16x8x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.float16x8x3_t [[TMP6]]
// CHECK-A32: ret void
float16x8x3_t test_vld1q_f16_x3(float16_t const *a) {
  return vld1q_f16_x3(a);
}

// CHECK-LABEL: @test_vld1q_f16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK-A32: %struct.float16x8x4_t* noalias sret(%struct.float16x8x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to [[HALF]]*
// CHECK: [[VLD1XN:%.*]] = call { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> } @llvm.{{aarch64.neon.ld1x4.v8f16.p0f16|arm.neon.vld1x4.v8i16.p0i16}}([[HALF]]* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }*
// CHECK: store { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> } [[VLD1XN]], { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x8x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float16x8x4_t, %struct.float16x8x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.float16x8x4_t [[TMP6]]
// CHECK-A32: ret void
float16x8x4_t test_vld1q_f16_x4(float16_t const *a) {
  return vld1q_f16_x4(a);
}

// CHECK-LABEL: @test_vld1q_f32_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK-A32: %struct.float32x4x2_t* noalias sret(%struct.float32x4x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float32x4x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK: [[VLD1XN:%.*]] = call { <4 x float>, <4 x float> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v4f32.p0f32(float* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x float>, <4 x float> }*
// CHECK: store { <4 x float>, <4 x float> } [[VLD1XN]], { <4 x float>, <4 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x4x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.float32x4x2_t [[TMP6]]
// CHECK-A32: ret void
float32x4x2_t test_vld1q_f32_x2(float32_t const *a) {
  return vld1q_f32_x2(a);
}

// CHECK-LABEL: @test_vld1q_f32_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK-A32: %struct.float32x4x3_t* noalias sret(%struct.float32x4x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float32x4x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK: [[VLD1XN:%.*]] = call { <4 x float>, <4 x float>, <4 x float> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v4f32.p0f32(float* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x float>, <4 x float>, <4 x float> }*
// CHECK: store { <4 x float>, <4 x float>, <4 x float> } [[VLD1XN]], { <4 x float>, <4 x float>, <4 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x4x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float32x4x3_t, %struct.float32x4x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.float32x4x3_t [[TMP6]]
// CHECK-A32: ret void
float32x4x3_t test_vld1q_f32_x3(float32_t const *a) {
  return vld1q_f32_x3(a);
}

// CHECK-LABEL: @test_vld1q_f32_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK-A32: %struct.float32x4x4_t* noalias sret(%struct.float32x4x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.float32x4x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK: [[VLD1XN:%.*]] = call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v4f32.p0f32(float* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x float>, <4 x float>, <4 x float>, <4 x float> }*
// CHECK: store { <4 x float>, <4 x float>, <4 x float>, <4 x float> } [[VLD1XN]], { <4 x float>, <4 x float>, <4 x float>, <4 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x4x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.float32x4x4_t, %struct.float32x4x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.float32x4x4_t [[TMP6]]
// CHECK-A32: ret void
float32x4x4_t test_vld1q_f32_x4(float32_t const *a) {
  return vld1q_f32_x4(a);
}

// CHECK-LABEL: @test_vld1q_p16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK-A32: %struct.poly16x8x2_t* noalias sret(%struct.poly16x8x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x8x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.poly16x8x2_t [[TMP6]]
// CHECK-A32: ret void
poly16x8x2_t test_vld1q_p16_x2(poly16_t const *a) {
  return vld1q_p16_x2(a);
}

// CHECK-LABEL: @test_vld1q_p16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK-A32: %struct.poly16x8x3_t* noalias sret(%struct.poly16x8x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x8x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.poly16x8x3_t [[TMP6]]
// CHECK-A32: ret void
poly16x8x3_t test_vld1q_p16_x3(poly16_t const *a) {
  return vld1q_p16_x3(a);
}

// CHECK-LABEL: @test_vld1q_p16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK-A32: %struct.poly16x8x4_t* noalias sret(%struct.poly16x8x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x8x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.poly16x8x4_t [[TMP6]]
// CHECK-A32: ret void
poly16x8x4_t test_vld1q_p16_x4(poly16_t const *a) {
  return vld1q_p16_x4(a);
}

// CHECK-LABEL: @test_vld1q_p8_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK-A32: %struct.poly8x16x2_t* noalias sret(%struct.poly8x16x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x16x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x16x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.poly8x16x2_t [[TMP4]]
// CHECK-A32: ret void
poly8x16x2_t test_vld1q_p8_x2(poly8_t const *a) {
  return vld1q_p8_x2(a);
}

// CHECK-LABEL: @test_vld1q_p8_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK-A32: %struct.poly8x16x3_t* noalias sret(%struct.poly8x16x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x16x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x16x3_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x16x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x16x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.poly8x16x3_t [[TMP4]]
// CHECK-A32: ret void
poly8x16x3_t test_vld1q_p8_x3(poly8_t const *a) {
  return vld1q_p8_x3(a);
}

// CHECK-LABEL: @test_vld1q_p8_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK-A32: %struct.poly8x16x4_t* noalias sret(%struct.poly8x16x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x16x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x16x4_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x16x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x16x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.poly8x16x4_t [[TMP4]]
// CHECK-A32: ret void
poly8x16x4_t test_vld1q_p8_x4(poly8_t const *a) {
  return vld1q_p8_x4(a);
}

// CHECK-LABEL: @test_vld1q_s16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK-A32: %struct.int16x8x2_t* noalias sret(%struct.int16x8x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x8x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int16x8x2_t [[TMP6]]
// CHECK-A32: ret void
int16x8x2_t test_vld1q_s16_x2(int16_t const *a) {
  return vld1q_s16_x2(a);
}

// CHECK-LABEL: @test_vld1q_s16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK-A32: %struct.int16x8x3_t* noalias sret(%struct.int16x8x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x8x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int16x8x3_t, %struct.int16x8x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int16x8x3_t [[TMP6]]
// CHECK-A32: ret void
int16x8x3_t test_vld1q_s16_x3(int16_t const *a) {
  return vld1q_s16_x3(a);
}

// CHECK-LABEL: @test_vld1q_s16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK-A32: %struct.int16x8x4_t* noalias sret(%struct.int16x8x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x8x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int16x8x4_t, %struct.int16x8x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int16x8x4_t [[TMP6]]
// CHECK-A32: ret void
int16x8x4_t test_vld1q_s16_x4(int16_t const *a) {
  return vld1q_s16_x4(a);
}

// CHECK-LABEL: @test_vld1q_s32_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK-A32: %struct.int32x4x2_t* noalias sret(%struct.int32x4x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int32x4x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v4i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32> } [[VLD1XN]], { <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x4x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int32x4x2_t [[TMP6]]
// CHECK-A32: ret void
int32x4x2_t test_vld1q_s32_x2(int32_t const *a) {
  return vld1q_s32_x2(a);
}

// CHECK-LABEL: @test_vld1q_s32_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK-A32: %struct.int32x4x3_t* noalias sret(%struct.int32x4x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int32x4x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v4i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD1XN]], { <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x4x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int32x4x3_t, %struct.int32x4x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int32x4x3_t [[TMP6]]
// CHECK-A32: ret void
int32x4x3_t test_vld1q_s32_x3(int32_t const *a) {
  return vld1q_s32_x3(a);
}

// CHECK-LABEL: @test_vld1q_s32_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK-A32: %struct.int32x4x4_t* noalias sret(%struct.int32x4x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int32x4x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v4i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD1XN]], { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x4x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int32x4x4_t, %struct.int32x4x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int32x4x4_t [[TMP6]]
// CHECK-A32: ret void
int32x4x4_t test_vld1q_s32_x4(int32_t const *a) {
  return vld1q_s32_x4(a);
}

// CHECK-LABEL: @test_vld1q_s64_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK-A32: %struct.int64x2x2_t* noalias sret(%struct.int64x2x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int64x2x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v2i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64> }*
// CHECK: store { <2 x i64>, <2 x i64> } [[VLD1XN]], { <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x2x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int64x2x2_t, %struct.int64x2x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int64x2x2_t [[TMP6]]
// CHECK-A32: ret void
int64x2x2_t test_vld1q_s64_x2(int64_t const *a) {
  return vld1q_s64_x2(a);
}

// CHECK-LABEL: @test_vld1q_s64_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK-A32: %struct.int64x2x3_t* noalias sret(%struct.int64x2x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int64x2x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v2i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK: store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x2x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int64x2x3_t, %struct.int64x2x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int64x2x3_t [[TMP6]]
// CHECK-A32: ret void
int64x2x3_t test_vld1q_s64_x3(int64_t const *a) {
  return vld1q_s64_x3(a);
}

// CHECK-LABEL: @test_vld1q_s64_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK-A32: %struct.int64x2x4_t* noalias sret(%struct.int64x2x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int64x2x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v2i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK: store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x2x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.int64x2x4_t, %struct.int64x2x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int64x2x4_t [[TMP6]]
// CHECK-A32: ret void
int64x2x4_t test_vld1q_s64_x4(int64_t const *a) {
  return vld1q_s64_x4(a);
}

// CHECK-LABEL: @test_vld1q_s8_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK-A32: %struct.int8x16x2_t* noalias sret(%struct.int8x16x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int8x16x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x16x2_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x16x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x16x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int8x16x2_t [[TMP4]]
// CHECK-A32: ret void
int8x16x2_t test_vld1q_s8_x2(int8_t const *a) {
  return vld1q_s8_x2(a);
}

// CHECK-LABEL: @test_vld1q_s8_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK-A32: %struct.int8x16x3_t* noalias sret(%struct.int8x16x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int8x16x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x16x3_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x16x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x16x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.int8x16x3_t, %struct.int8x16x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int8x16x3_t [[TMP4]]
// CHECK-A32: ret void
int8x16x3_t test_vld1q_s8_x3(int8_t const *a) {
  return vld1q_s8_x3(a);
}

// CHECK-LABEL: @test_vld1q_s8_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK-A32: %struct.int8x16x4_t* noalias sret(%struct.int8x16x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.int8x16x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x16x4_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x16x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x16x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.int8x16x4_t, %struct.int8x16x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.int8x16x4_t [[TMP4]]
// CHECK-A32: ret void
int8x16x4_t test_vld1q_s8_x4(int8_t const *a) {
  return vld1q_s8_x4(a);
}

// CHECK-LABEL: @test_vld1q_u16_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK-A32: %struct.uint16x8x2_t* noalias sret(%struct.uint16x8x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x8x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint16x8x2_t [[TMP6]]
// CHECK-A32: ret void
uint16x8x2_t test_vld1q_u16_x2(uint16_t const *a) {
  return vld1q_u16_x2(a);
}

// CHECK-LABEL: @test_vld1q_u16_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK-A32: %struct.uint16x8x3_t* noalias sret(%struct.uint16x8x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x8x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint16x8x3_t [[TMP6]]
// CHECK-A32: ret void
uint16x8x3_t test_vld1q_u16_x3(uint16_t const *a) {
  return vld1q_u16_x3(a);
}

// CHECK-LABEL: @test_vld1q_u16_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK-A32: %struct.uint16x8x4_t* noalias sret(%struct.uint16x8x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK: [[VLD1XN:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v8i16.p0i16(i16* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD1XN]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x8x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint16x8x4_t [[TMP6]]
// CHECK-A32: ret void
uint16x8x4_t test_vld1q_u16_x4(uint16_t const *a) {
  return vld1q_u16_x4(a);
}

// CHECK-LABEL: @test_vld1q_u32_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK-A32: %struct.uint32x4x2_t* noalias sret(%struct.uint32x4x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x4x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v4i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32> } [[VLD1XN]], { <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x4x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint32x4x2_t [[TMP6]]
// CHECK-A32: ret void
uint32x4x2_t test_vld1q_u32_x2(uint32_t const *a) {
  return vld1q_u32_x2(a);
}

// CHECK-LABEL: @test_vld1q_u32_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK-A32: %struct.uint32x4x3_t* noalias sret(%struct.uint32x4x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x4x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v4i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD1XN]], { <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x4x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint32x4x3_t [[TMP6]]
// CHECK-A32: ret void
uint32x4x3_t test_vld1q_u32_x3(uint32_t const *a) {
  return vld1q_u32_x3(a);
}

// CHECK-LABEL: @test_vld1q_u32_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK-A32: %struct.uint32x4x4_t* noalias sret(%struct.uint32x4x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x4x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK: [[VLD1XN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v4i32.p0i32(i32* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD1XN]], { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x4x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint32x4x4_t [[TMP6]]
// CHECK-A32: ret void
uint32x4x4_t test_vld1q_u32_x4(uint32_t const *a) {
  return vld1q_u32_x4(a);
}

// CHECK-LABEL: @test_vld1q_u64_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK-A32: %struct.uint64x2x2_t* noalias sret(%struct.uint64x2x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x2x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v2i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64> }*
// CHECK: store { <2 x i64>, <2 x i64> } [[VLD1XN]], { <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x2x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint64x2x2_t [[TMP6]]
// CHECK-A32: ret void
uint64x2x2_t test_vld1q_u64_x2(uint64_t const *a) {
  return vld1q_u64_x2(a);
}

// CHECK-LABEL: @test_vld1q_u64_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK-A32: %struct.uint64x2x3_t* noalias sret(%struct.uint64x2x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x2x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v2i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK: store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x2x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint64x2x3_t [[TMP6]]
// CHECK-A32: ret void
uint64x2x3_t test_vld1q_u64_x3(uint64_t const *a) {
  return vld1q_u64_x3(a);
}

// CHECK-LABEL: @test_vld1q_u64_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK-A32: %struct.uint64x2x4_t* noalias sret(%struct.uint64x2x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x2x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK: [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v2i64.p0i64(i64* [[TMP2]])
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK: store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x2x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP6:%.*]] = load %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint64x2x4_t [[TMP6]]
// CHECK-A32: ret void
uint64x2x4_t test_vld1q_u64_x4(uint64_t const *a) {
  return vld1q_u64_x4(a);
}

// CHECK-LABEL: @test_vld1q_u8_x2(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK-A32: %struct.uint8x16x2_t* noalias sret(%struct.uint8x16x2_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x16x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x2|arm.neon.vld1x2}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x16x2_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint8x16x2_t [[TMP4]]
// CHECK-A32: ret void
uint8x16x2_t test_vld1q_u8_x2(uint8_t const *a) {
  return vld1q_u8_x2(a);
}

// CHECK-LABEL: @test_vld1q_u8_x3(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK-A32: %struct.uint8x16x3_t* noalias sret(%struct.uint8x16x3_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x16x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x16x3_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x3|arm.neon.vld1x3}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x16x3_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x16x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 48, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint8x16x3_t [[TMP4]]
// CHECK-A32: ret void
uint8x16x3_t test_vld1q_u8_x3(uint8_t const *a) {
  return vld1q_u8_x3(a);
}

// CHECK-LABEL: @test_vld1q_u8_x4(
// CHECK-A64: [[RETVAL:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK-A32: %struct.uint8x16x4_t* noalias sret(%struct.uint8x16x4_t) align 8 [[RETVAL:%.*]],
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x16x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x16x4_t* [[__RET]] to i8*
// CHECK: [[VLD1XN:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.{{aarch64.neon.ld1x4|arm.neon.vld1x4}}.v16i8.p0i8(i8* %a)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD1XN]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x16x4_t* [[RETVAL]] to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x16x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 64, i1 false)
// CHECK-A64: [[TMP4:%.*]] = load %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[RETVAL]], align 16
// CHECK-A64: ret %struct.uint8x16x4_t [[TMP4]]
// CHECK-A32: ret void
uint8x16x4_t test_vld1q_u8_x4(uint8_t const *a) {
  return vld1q_u8_x4(a);
}

// CHECK-LABEL: @test_vld2_dup_f16(
// CHECK: [[__RET:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to half*
// CHECK-A64: [[VLD2:%.*]] = call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2r.v4f16.p0f16(half* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x [[HALF]]>, <4 x [[HALF]]> }*
// CHECK: store { <4 x [[HALF]]>, <4 x [[HALF]]> } [[VLD2]], { <4 x [[HALF]]>, <4 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x4x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_f16(float16x4x2_t *dest, const float16_t *src) {
  *dest = vld2_dup_f16(src);
}

// CHECK-LABEL: @test_vld2_dup_f32(
// CHECK: [[__RET:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK-A64: [[VLD2:%.*]] = call { <2 x float>, <2 x float> } @llvm.aarch64.neon.ld2r.v2f32.p0f32(float* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <2 x float>, <2 x float> } @llvm.arm.neon.vld2dup.v2f32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x float>, <2 x float> }*
// CHECK: store { <2 x float>, <2 x float> } [[VLD2]], { <2 x float>, <2 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x2x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_f32(float32x2x2_t *dest, const float32_t *src) {
  *dest = vld2_dup_f32(src);
}

// CHECK-LABEL: @test_vld2_dup_p16(
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16> } [[VLD2]], { <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x4x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_p16(poly16x4x2_t *dest, const poly16_t *src) {
  *dest = vld2_dup_p16(src);
}

// CHECK-LABEL: @test_vld2_dup_p8(
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET]] to i8*
// CHECK-A64: [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8> } [[VLD2]], { <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x8x2_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_p8(poly8x8x2_t *dest, poly8_t *src) {
  *dest = vld2_dup_p8(src);
}

// CHECK-LABEL: @test_vld2_dup_s16(
// CHECK: [[__RET:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16> } [[VLD2]], { <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x4x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_s16(int16x4x2_t *dest, const int16_t *src) {
  *dest = vld2_dup_s16(src);
}

// CHECK-LABEL: @test_vld2_dup_s32(
// CHECK: [[__RET:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2r.v2i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2dup.v2i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32> } [[VLD2]], { <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x2x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_s32(int32x2x2_t *dest, const int32_t *src) {
  *dest = vld2_dup_s32(src);
}

// CHECK-LABEL: @test_vld2_dup_s8(
// CHECK: [[__RET:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x8x2_t* [[__RET]] to i8*
// CHECK-A64: [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8> } [[VLD2]], { <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x8x2_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_s8(int8x8x2_t *dest, int8_t *src) {
  *dest = vld2_dup_s8(src);
}

// CHECK-LABEL: @test_vld2_dup_u16(
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16> } [[VLD2]], { <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x4x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_u16(uint16x4x2_t *dest, const uint16_t *src) {
  *dest = vld2_dup_u16(src);
}

// CHECK-LABEL: @test_vld2_dup_u32(
// CHECK: entry:
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2r.v2i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2dup.v2i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32> } [[VLD2]], { <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x2x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_u32(uint32x2x2_t *dest, const uint32_t *src) {
  *dest = vld2_dup_u32(src);
}

// CHECK-LABEL: @test_vld2_dup_s64(
// CHECK: [[__RET:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x1x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK-A64: [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2r.v1i64.p0i64(i64* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2dup.v1i64.p0i8(i8* [[TMP1]], i32 8)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64> } [[VLD2]], { <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x1x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x1x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_s64(int64x1x2_t *dest, const int64_t *src) {
  *dest = vld2_dup_s64(src);
}

// CHECK-LABEL: @test_vld2_dup_u64(
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x1x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK-A64: [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2r.v1i64.p0i64(i64* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2dup.v1i64.p0i8(i8* [[TMP1]], i32 8)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64> } [[VLD2]], { <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x1x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x1x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_u64(uint64x1x2_t *dest, const uint64_t *src) {
  *dest = vld2_dup_u64(src);
}

// CHECK-LABEL: @test_vld2_dup_u8(
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET]] to i8*
// CHECK-A64: [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8> } [[VLD2]], { <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x8x2_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 16, i1 false)
// CHECK: ret void
void test_vld2_dup_u8(uint8x8x2_t *dest, const uint8_t *src) {
  *dest = vld2_dup_u8(src);
}

// CHECK-LABEL: @test_vld3_dup_f16(
// CHECK: [[__RET:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to half*
// CHECK-A64: [[VLD3:%.*]] = call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3r.v4f16.p0f16(half* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }*
// CHECK: store { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> } [[VLD3]], { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x4x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_f16(float16x4x3_t *dest, float16_t *src) {
  *dest = vld3_dup_f16(src);
}

// CHECK-LABEL: @test_vld3_dup_f32(
// CHECK: [[__RET:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK-A64: [[VLD3:%.*]] = call { <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld3r.v2f32.p0f32(float* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3dup.v2f32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x float>, <2 x float>, <2 x float> }*
// CHECK: store { <2 x float>, <2 x float>, <2 x float> } [[VLD3]], { <2 x float>, <2 x float>, <2 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x2x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_f32(float32x2x3_t *dest, const float32_t *src) {
  *dest = vld3_dup_f32(src);
}

// CHECK-LABEL: @test_vld3_dup_p16(
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x4x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_p16(poly16x4x3_t *dest, const poly16_t *src) {
  *dest = vld3_dup_p16(src);
}

// CHECK-LABEL: @test_vld3_dup_p8(
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x8x3_t* [[__RET]] to i8*
// CHECK-A64: [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x8x3_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_p8(poly8x8x3_t *dest, const poly8_t *src) {
  *dest = vld3_dup_p8(src);
}

// CHECK-LABEL: @test_vld3_dup_s16(
// CHECK: [[__RET:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x4x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_s16(int16x4x3_t *dest, const int16_t *src) {
  *dest = vld3_dup_s16(src);
}

// CHECK-LABEL: @test_vld3_dup_s32(
// CHECK: [[__RET:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3r.v2i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3dup.v2i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3]], { <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x2x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_s32(int32x2x3_t *dest, const int32_t *src) {
  *dest = vld3_dup_s32(src);
}

// CHECK-LABEL: @test_vld3_dup_s8(
// CHECK: [[__RET:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x8x3_t* [[__RET]] to i8*
// CHECK-A64: [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x8x3_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_s8(int8x8x3_t *dest, const int8_t *src) {
  *dest = vld3_dup_s8(src);
}

// CHECK-LABEL: @test_vld3_dup_u16(
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x4x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_u16(uint16x4x3_t *dest, const uint16_t *src) {
  *dest = vld3_dup_u16(src);
}

// CHECK-LABEL: @test_vld3_dup_u32(
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x2x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3r.v2i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3dup.v2i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3]], { <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x2x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x2x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_u32(uint32x2x3_t *dest, const uint32_t *src) {
  *dest = vld3_dup_u32(src);
}

// CHECK-LABEL: @test_vld3_dup_u8(
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x8x3_t* [[__RET]] to i8*
// CHECK-A64: [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x8x3_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_u8(uint8x8x3_t *dest, const uint8_t *src) {
  *dest = vld3_dup_u8(src);
}

// CHECK-LABEL: @test_vld3_dup_s64(
// CHECK: [[__RET:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x1x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK-A64: [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3r.v1i64.p0i64(i64* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3dup.v1i64.p0i8(i8* [[TMP1]], i32 8)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x1x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x1x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_s64(int64x1x3_t *dest, const int64_t *src) {
  *dest = vld3_dup_s64(src);
}

// CHECK-LABEL: @test_vld3_dup_u64(
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x1x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK-A64: [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3r.v1i64.p0i64(i64* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3dup.v1i64.p0i8(i8* [[TMP1]], i32 8)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x1x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x1x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 24, i1 false)
// CHECK: ret void
void test_vld3_dup_u64(uint64x1x3_t *dest, const uint64_t *src) {
  *dest = vld3_dup_u64(src);
}

// CHECK-LABEL: @test_vld4_dup_f16(
// CHECK: [[__RET:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to half*
// CHECK-A64: [[VLD4:%.*]] = call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4r.v4f16.p0f16(half* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }*
// CHECK: store { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> } [[VLD4]], { <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]>, <4 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x4x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_f16(float16x4x4_t *dest, const float16_t *src) {
  *dest = vld4_dup_f16(src);
}

// CHECK-LABEL: @test_vld4_dup_f32(
// CHECK: [[__RET:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK-A64: [[VLD4:%.*]] = call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld4r.v2f32.p0f32(float* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4dup.v2f32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x float>, <2 x float>, <2 x float>, <2 x float> }*
// CHECK: store { <2 x float>, <2 x float>, <2 x float>, <2 x float> } [[VLD4]], { <2 x float>, <2 x float>, <2 x float>, <2 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x2x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_f32(float32x2x4_t *dest, const float32_t *src) {
  *dest = vld4_dup_f32(src);
}

// CHECK-LABEL: @test_vld4_dup_p16(
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x4x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_p16(poly16x4x4_t *dest, const poly16_t *src) {
  *dest = vld4_dup_p16(src);
}

// CHECK-LABEL: @test_vld4_dup_p8(
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x8x4_t* [[__RET]] to i8*
// CHECK-A64: [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x8x4_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_p8(poly8x8x4_t *dest, const poly8_t *src) {
  *dest = vld4_dup_p8(src);
}

// CHECK-LABEL: @test_vld4_dup_s16(
// CHECK: [[__RET:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x4x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_s16(int16x4x4_t *dest, const int16_t *src) {
  *dest = vld4_dup_s16(src);
}

// CHECK-LABEL: @test_vld4_dup_s32(
// CHECK: [[__RET:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4r.v2i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4dup.v2i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4]], { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x2x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_s32(int32x2x4_t *dest, const int32_t *src) {
  *dest = vld4_dup_s32(src);
}

// CHECK-LABEL: @test_vld4_dup_s8(
// CHECK: [[__RET:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x8x4_t* [[__RET]] to i8*
// CHECK-A64: [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x8x4_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_s8(int8x8x4_t *dest, const int8_t *src) {
  *dest = vld4_dup_s8(src);
}

// CHECK-LABEL: @test_vld4_dup_u16(
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4r.v4i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4dup.v4i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK: store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x4x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_u16(uint16x4x4_t *dest, const uint16_t *src) {
  *dest = vld4_dup_u16(src);
}

// CHECK-LABEL: @test_vld4_dup_u32(
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x2x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4r.v2i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4dup.v2i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK: store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4]], { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x2x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x2x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_u32(uint32x2x4_t *dest, const uint32_t *src) {
  *dest = vld4_dup_u32(src);
}

// CHECK-LABEL: @test_vld4_dup_u8(
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x8x4_t* [[__RET]] to i8*
// CHECK-A64: [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4r.v8i8.p0i8(i8* %src)
// CHECK-A32: [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4dup.v8i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK: store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x8x4_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP2]], i8* align 8 [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_u8(uint8x8x4_t *dest, const uint8_t *src) {
  *dest = vld4_dup_u8(src);
}

// CHECK-LABEL: @test_vld4_dup_s64(
// CHECK: [[__RET:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.int64x1x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK-A64: [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4r.v1i64.p0i64(i64* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4dup.v1i64.p0i8(i8* [[TMP1]], i32 8)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int64x1x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int64x1x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_s64(int64x1x4_t *dest, const int64_t *src) {
  *dest = vld4_dup_s64(src);
}

// CHECK-LABEL: @test_vld4_dup_u64(
// CHECK: [[__RET:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint64x1x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i64* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK-A64: [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4r.v1i64.p0i64(i64* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4dup.v1i64.p0i8(i8* [[TMP1]], i32 8)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK: store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint64x1x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint64x1x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld4_dup_u64(uint64x1x4_t *dest, const uint64_t *src) {
  *dest = vld4_dup_u64(src);
}

// CHECK-LABEL: @test_vld2q_dup_f16(
// CHECK: [[__RET:%.*]] = alloca %struct.float16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to half*
// CHECK-A64: [[VLD2:%.*]] = call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2r.v8f16.p0f16(half* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x [[HALF]]>, <8 x [[HALF]]> }*
// CHECK: store { <8 x [[HALF]]>, <8 x [[HALF]]> } [[VLD2]], { <8 x [[HALF]]>, <8 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x8x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_f16(float16x8x2_t *dest, const float16_t *src) {
  *dest = vld2q_dup_f16(src);
}

// CHECK-LABEL: @test_vld2q_dup_f32(
// CHECK: [[__RET:%.*]] = alloca %struct.float32x4x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK-A64: [[VLD2:%.*]] = call { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld2r.v4f32.p0f32(float* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2dup.v4f32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x float>, <4 x float> }*
// CHECK: store { <4 x float>, <4 x float> } [[VLD2]], { <4 x float>, <4 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x4x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_f32(float32x4x2_t *dest, const float32_t *src) {
  *dest = vld2q_dup_f32(src);
}

// CHECK-LABEL: @test_vld2q_dup_p16(
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16> } [[VLD2]], { <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x8x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_p16(poly16x8x2_t *dest, const poly16_t *src) {
  *dest = vld2q_dup_p16(src);
}

// CHECK-LABEL: @test_vld2q_dup_p8(
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x16x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET]] to i8*
// CHECK-A64: [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8> } [[VLD2]], { <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x16x2_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_p8(poly8x16x2_t *dest, const poly8_t *src) {
  *dest = vld2q_dup_p8(src);
}

// CHECK-LABEL: @test_vld2q_dup_s16(
// CHECK: [[__RET:%.*]] = alloca %struct.int16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16> } [[VLD2]], { <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x8x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_s16(int16x8x2_t *dest, const int16_t *src) {
  *dest = vld2q_dup_s16(src);
}

// CHECK-LABEL: @test_vld2q_dup_s32(
// CHECK: [[__RET:%.*]] = alloca %struct.int32x4x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2r.v4i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2dup.v4i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32> } [[VLD2]], { <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x4x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_s32(int32x4x2_t *dest, const int32_t  *src) {
  *dest = vld2q_dup_s32(src);
}

// CHECK-LABEL: @test_vld2q_dup_s8(
// CHECK: [[__RET:%.*]] = alloca %struct.int8x16x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x16x2_t* [[__RET]] to i8*
// CHECK-A64: [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8> } [[VLD2]], { <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x16x2_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x16x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_s8(int8x16x2_t *dest, const int8_t *src) {
  *dest = vld2q_dup_s8(src);
}

// CHECK-LABEL: @test_vld2q_dup_u16(
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x8x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16> } [[VLD2]], { <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x8x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_u16(uint16x8x2_t *dest, const uint16_t *src) {
  *dest = vld2q_dup_u16(src);
}

// CHECK-LABEL: @test_vld2q_dup_u32(
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x4x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2r.v4i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2dup.v4i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32> } [[VLD2]], { <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x4x2_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_u32(uint32x4x2_t *dest, const uint32_t *src) {
  *dest = vld2q_dup_u32(src);
}

// CHECK-LABEL: @test_vld2q_dup_u8(
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x16x2_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET]] to i8*
// CHECK-A64: [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8> } [[VLD2]], { <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x16x2_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 32, i1 false)
// CHECK: ret void
void test_vld2q_dup_u8(uint8x16x2_t *dest, const uint8_t *src) {
  *dest = vld2q_dup_u8(src);
}

// CHECK-LABEL: @test_vld3q_dup_f16(
// CHECK: [[__RET:%.*]] = alloca %struct.float16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to half*
// CHECK-A64: [[VLD3:%.*]] = call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3r.v8f16.p0f16(half* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }*
// CHECK: store { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> } [[VLD3]], { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x8x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_f16(float16x8x3_t *dest, const float16_t *src) {
  *dest = vld3q_dup_f16(src);
}

// CHECK-LABEL: @test_vld3q_dup_f32(
// CHECK: [[__RET:%.*]] = alloca %struct.float32x4x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK-A64: [[VLD3:%.*]] = call { <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld3r.v4f32.p0f32(float* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3dup.v4f32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x float>, <4 x float>, <4 x float> }*
// CHECK: store { <4 x float>, <4 x float>, <4 x float> } [[VLD3]], { <4 x float>, <4 x float>, <4 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x4x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_f32(float32x4x3_t *dest, const float32_t *src) {
  *dest = vld3q_dup_f32(src);
}

// CHECK-LABEL: @test_vld3q_dup_p16(
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x8x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_p16(poly16x8x3_t *dest, const poly16_t *src) {
  *dest = vld3q_dup_p16(src);
}

// CHECK-LABEL: @test_vld3q_dup_p8(
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x16x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x16x3_t* [[__RET]] to i8*
// CHECK-A64: [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x16x3_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x16x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_p8(poly8x16x3_t *dest, const poly8_t *src) {
  *dest = vld3q_dup_p8(src);
}

// CHECK-LABEL: @test_vld3q_dup_s16(
// CHECK: [[__RET:%.*]] = alloca %struct.int16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x8x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_s16(int16x8x3_t *dest, const int16_t *src) {
  *dest = vld3q_dup_s16(src);
}

// CHECK-LABEL: @test_vld3q_dup_s32(
// CHECK: [[__RET:%.*]] = alloca %struct.int32x4x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3r.v4i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3dup.v4i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3]], { <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x4x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_s32(int32x4x3_t *dest, const int32_t *src) {
  *dest = vld3q_dup_s32(src);
}

// CHECK-LABEL: @test_vld3q_dup_s8(
// CHECK: [[__RET:%.*]] = alloca %struct.int8x16x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x16x3_t* [[__RET]] to i8*
// CHECK-A64: [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x16x3_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x16x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_s8(int8x16x3_t *dest, const int8_t *src) {
  *dest = vld3q_dup_s8(src);
}

// CHECK-LABEL: @test_vld3q_dup_u16(
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x8x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x8x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x8x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x8x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_u16(uint16x8x3_t *dest, const uint16_t *src) {
  *dest = vld3q_dup_u16(src);
}

// CHECK-LABEL: @test_vld3q_dup_u32(
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x4x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x4x3_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3r.v4i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3dup.v4i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3]], { <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x4x3_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x4x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_u32(uint32x4x3_t *dest, const uint32_t *src) {
  *dest = vld3q_dup_u32(src);
}

// CHECK-LABEL: @test_vld3q_dup_u8(
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x16x3_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x16x3_t* [[__RET]] to i8*
// CHECK-A64: [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x16x3_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x16x3_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 48, i1 false)
// CHECK: ret void
void test_vld3q_dup_u8(uint8x16x3_t *dest, const uint8_t *src) {
  *dest = vld3q_dup_u8(src);
}

// CHECK-LABEL: @test_vld4q_dup_f16(
// CHECK: [[__RET:%.*]] = alloca %struct.float16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast half* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to half*
// CHECK-A64: [[VLD4:%.*]] = call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4r.v8f16.p0f16(half* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }*
// CHECK: store { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> } [[VLD4]], { <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]>, <8 x [[HALF]]> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float16x8x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_f16(float16x8x4_t *dest, const float16_t *src) {
  *dest = vld4q_dup_f16(src);
}

// CHECK-LABEL: @test_vld4q_dup_f32(
// CHECK: [[__RET:%.*]] = alloca %struct.float32x4x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.float32x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast float* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to float*
// CHECK-A64: [[VLD4:%.*]] = call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld4r.v4f32.p0f32(float* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4dup.v4f32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x float>, <4 x float>, <4 x float>, <4 x float> }*
// CHECK: store { <4 x float>, <4 x float>, <4 x float>, <4 x float> } [[VLD4]], { <4 x float>, <4 x float>, <4 x float>, <4 x float> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.float32x4x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.float32x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_f32(float32x4x4_t *dest, const float32_t *src) {
  *dest = vld4q_dup_f32(src);
}

// CHECK-LABEL: @test_vld4q_dup_p16(
// CHECK: [[__RET:%.*]] = alloca %struct.poly16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.poly16x8x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.poly16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_p16(poly16x8x4_t *dest, const poly16_t *src) {
  *dest = vld4q_dup_p16(src);
}

// CHECK-LABEL: @test_vld4q_dup_p8(
// CHECK: [[__RET:%.*]] = alloca %struct.poly8x16x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.poly8x16x4_t* [[__RET]] to i8*
// CHECK-A64: [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.poly8x16x4_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.poly8x16x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_p8(poly8x16x4_t *dest, const poly8_t *src) {
  *dest = vld4q_dup_p8(src);
}

// CHECK-LABEL: @test_vld4q_dup_s16(
// CHECK: [[__RET:%.*]] = alloca %struct.int16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int16x8x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_s16(int16x8x4_t *dest, const int16_t *src) {
  *dest = vld4q_dup_s16(src);
}

// CHECK-LABEL: @test_vld4q_dup_s32(
// CHECK: [[__RET:%.*]] = alloca %struct.int32x4x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int32x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4r.v4i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4dup.v4i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4]], { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.int32x4x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.int32x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_s32(int32x4x4_t *dest, const int32_t *src) {
  *dest = vld4q_dup_s32(src);
}

// CHECK-LABEL: @test_vld4q_dup_s8(
// CHECK: [[__RET:%.*]] = alloca %struct.int8x16x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.int8x16x4_t* [[__RET]] to i8*
// CHECK-A64: [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.int8x16x4_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.int8x16x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_s8(int8x16x4_t *dest, const int8_t *src) {
  *dest = vld4q_dup_s8(src);
}

// CHECK-LABEL: @test_vld4q_dup_u16(
// CHECK: [[__RET:%.*]] = alloca %struct.uint16x8x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint16x8x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i16* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i16*
// CHECK-A64: [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4r.v8i16.p0i16(i16* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4dup.v8i16.p0i8(i8* [[TMP1]], i32 2)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK: store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint16x8x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint16x8x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_u16(uint16x8x4_t *dest, const uint16_t *src) {
  *dest = vld4q_dup_u16(src);
}

// CHECK-LABEL: @test_vld4q_dup_u32(
// CHECK: [[__RET:%.*]] = alloca %struct.uint32x4x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint32x4x4_t* [[__RET]] to i8*
// CHECK: [[TMP1:%.*]] = bitcast i32* %src to i8*
// CHECK-A64: [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i32*
// CHECK-A64: [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4r.v4i32.p0i32(i32* [[TMP2]])
// CHECK-A32: [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4dup.v4i32.p0i8(i8* [[TMP1]], i32 4)
// CHECK: [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK: store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4]], { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP3]]
// CHECK: [[TMP4:%.*]] = bitcast %struct.uint32x4x4_t* %dest to i8*
// CHECK: [[TMP5:%.*]] = bitcast %struct.uint32x4x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP4]], i8* align {{16|8}} [[TMP5]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_u32(uint32x4x4_t *dest, const uint32_t *src) {
  *dest = vld4q_dup_u32(src);
}

// CHECK-LABEL: @test_vld4q_dup_u8(
// CHECK: [[__RET:%.*]] = alloca %struct.uint8x16x4_t, align {{16|8}}
// CHECK: [[TMP0:%.*]] = bitcast %struct.uint8x16x4_t* [[__RET]] to i8*
// CHECK-A64: [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4r.v16i8.p0i8(i8* %src)
// CHECK-A32: [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4dup.v16i8.p0i8(i8* %src, i32 1)
// CHECK: [[TMP1:%.*]] = bitcast i8* [[TMP0]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK: store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP1]]
// CHECK: [[TMP2:%.*]] = bitcast %struct.uint8x16x4_t* %dest to i8*
// CHECK: [[TMP3:%.*]] = bitcast %struct.uint8x16x4_t* [[__RET]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.{{i64|i32}}(i8* align {{16|8}} [[TMP2]], i8* align {{16|8}} [[TMP3]], {{i64|i32}} 64, i1 false)
// CHECK: ret void
void test_vld4q_dup_u8(uint8x16x4_t *dest, const uint8_t *src) {
  *dest = vld4q_dup_u8(src);
}
