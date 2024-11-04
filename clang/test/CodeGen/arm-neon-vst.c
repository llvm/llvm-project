// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:     -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | \
// RUN:     FileCheck -check-prefixes=CHECK,CHECK-A64 %s
// RUN: %clang_cc1 -triple armv8-none-linux-gnueabi -target-feature +neon \
// RUN:     -target-feature +fp16 -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:     opt -S -passes=mem2reg | FileCheck -check-prefixes=CHECK,CHECK-A32 %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: @test_vst1_f16_x2(
// CHECK: [[B:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x half>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x half>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x [[HALF:(half|i16)]]>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x [[HALF]]>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v4f16.p0(<4 x half> [[TMP7]], <4 x half> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v4i16(ptr %a, <4 x i16> [[TMP7]], <4 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1_f16_x2(float16_t *a, float16x4x2_t b) {
  vst1_f16_x2(a, b);
}

// CHECK-LABEL: @test_vst1_f16_x3(
// CHECK: [[B:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x half>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x [[HALF]]>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x [[HALF]]>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x [[HALF]]>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v4f16.p0(<4 x half> [[TMP9]], <4 x half> [[TMP10]], <4 x half> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v4i16(ptr %a, <4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1_f16_x3(float16_t *a, float16x4x3_t b) {
  vst1_f16_x3(a, b);
}

// CHECK-LABEL: @test_vst1_f16_x4(
// CHECK: [[B:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x half>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <4 x half>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <4 x half> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x [[HALF]]>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x [[HALF]]>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x [[HALF]]>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x [[HALF]]>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v4f16.p0(<4 x half> [[TMP11]], <4 x half> [[TMP12]], <4 x half> [[TMP13]], <4 x half> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v4i16(ptr %a, <4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1_f16_x4(float16_t *a, float16x4x4_t b) {
  vst1_f16_x4(a, b);
}

// CHECK-LABEL: @test_vst1_f32_x2(
// CHECK: [[B:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x float>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x float>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v2f32.p0(<2 x float> [[TMP7]], <2 x float> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v2f32(ptr %a, <2 x float> [[TMP7]], <2 x float> [[TMP8]])
// CHECK: ret void
void test_vst1_f32_x2(float32_t *a, float32x2x2_t b) {
  vst1_f32_x2(a, b);
}

// CHECK-LABEL: @test_vst1_f32_x3(
// CHECK: [[B:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x float>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v2f32.p0(<2 x float> [[TMP9]], <2 x float> [[TMP10]], <2 x float> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v2f32(ptr %a, <2 x float> [[TMP9]], <2 x float> [[TMP10]], <2 x float> [[TMP11]])
// CHECK: ret void
void test_vst1_f32_x3(float32_t *a, float32x2x3_t b) {
  vst1_f32_x3(a, b);
}

// CHECK-LABEL: @test_vst1_f32_x4(
// CHECK: [[B:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x float>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <2 x float>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <2 x float> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x float>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v2f32.p0(<2 x float> [[TMP11]], <2 x float> [[TMP12]], <2 x float> [[TMP13]], <2 x float> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v2f32(ptr %a, <2 x float> [[TMP11]], <2 x float> [[TMP12]], <2 x float> [[TMP13]], <2 x float> [[TMP14]])
// CHECK: ret void
void test_vst1_f32_x4(float32_t *a, float32x2x4_t b) {
  vst1_f32_x4(a, b);
}

// CHECK-LABEL: @test_vst1_p16_x2(
// CHECK: [[B:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v4i16(ptr %a, <4 x i16> [[TMP7]], <4 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1_p16_x2(poly16_t *a, poly16x4x2_t b) {
  vst1_p16_x2(a, b);
}

// CHECK-LABEL: @test_vst1_p16_x3(
// CHECK: [[B:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v4i16(ptr %a, <4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1_p16_x3(poly16_t *a, poly16x4x3_t b) {
  vst1_p16_x3(a, b);
}

// CHECK-LABEL: @test_vst1_p16_x4(
// CHECK: [[B:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v4i16(ptr %a, <4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1_p16_x4(poly16_t *a, poly16x4x4_t b) {
  vst1_p16_x4(a, b);
}

// CHECK-LABEL: @test_vst1_p8_x2(
// CHECK: [[B:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]])
// CHECK: ret void
void test_vst1_p8_x2(poly8_t *a, poly8x8x2_t b) {
  vst1_p8_x2(a, b);
}

// CHECK-LABEL: @test_vst1_p8_x3(
// CHECK: [[B:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]])
// CHECK: ret void
void test_vst1_p8_x3(poly8_t *a, poly8x8x3_t b) {
  vst1_p8_x3(a, b);
}

// CHECK-LABEL: @test_vst1_p8_x4(
// CHECK: [[B:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]])
// CHECK: ret void
void test_vst1_p8_x4(poly8_t *a, poly8x8x4_t b) {
  vst1_p8_x4(a, b);
}

// CHECK-LABEL: @test_vst1_s16_x2(
// CHECK: [[B:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v4i16(ptr %a, <4 x i16> [[TMP7]], <4 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1_s16_x2(int16_t *a, int16x4x2_t b) {
  vst1_s16_x2(a, b);
}

// CHECK-LABEL: @test_vst1_s16_x3(
// CHECK: [[B:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v4i16(ptr %a, <4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1_s16_x3(int16_t *a, int16x4x3_t b) {
  vst1_s16_x3(a, b);
}

// CHECK-LABEL: @test_vst1_s16_x4(
// CHECK: [[B:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v4i16(ptr %a, <4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1_s16_x4(int16_t *a, int16x4x4_t b) {
  vst1_s16_x4(a, b);
}

// CHECK-LABEL: @test_vst1_s32_x2(
// CHECK: [[B:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v2i32.p0(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v2i32(ptr %a, <2 x i32> [[TMP7]], <2 x i32> [[TMP8]])
// CHECK: ret void
void test_vst1_s32_x2(int32_t *a, int32x2x2_t b) {
  vst1_s32_x2(a, b);
}

// CHECK-LABEL: @test_vst1_s32_x3(
// CHECK: [[B:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v2i32.p0(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v2i32(ptr %a, <2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]])
// CHECK: ret void
void test_vst1_s32_x3(int32_t *a, int32x2x3_t b) {
  vst1_s32_x3(a, b);
}

// CHECK-LABEL: @test_vst1_s32_x4(
// CHECK: [[B:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <2 x i32>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v2i32.p0(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v2i32(ptr %a, <2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]])
// CHECK: ret void
void test_vst1_s32_x4(int32_t *a, int32x2x4_t b) {
  vst1_s32_x4(a, b);
}

// CHECK-LABEL: @test_vst1_s64_x2(
// CHECK: [[B:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x1x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v1i64(ptr %a, <1 x i64> [[TMP7]], <1 x i64> [[TMP8]])
// CHECK: ret void
void test_vst1_s64_x2(int64_t *a, int64x1x2_t b) {
  vst1_s64_x2(a, b);
}

// CHECK-LABEL: @test_vst1_s64_x3(
// CHECK: [[B:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v1i64(ptr %a, <1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]])
// CHECK: ret void
void test_vst1_s64_x3(int64_t *a, int64x1x3_t b) {
  vst1_s64_x3(a, b);
}

// CHECK-LABEL: @test_vst1_s64_x4(
// CHECK: [[B:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v1i64(ptr %a, <1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]])
// CHECK: ret void
void test_vst1_s64_x4(int64_t *a, int64x1x4_t b) {
  vst1_s64_x4(a, b);
}

// CHECK-LABEL: @test_vst1_s8_x2(
// CHECK: [[B:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]])
// CHECK: ret void
void test_vst1_s8_x2(int8_t *a, int8x8x2_t b) {
  vst1_s8_x2(a, b);
}

// CHECK-LABEL: @test_vst1_s8_x3(
// CHECK: [[B:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]])
// CHECK: ret void
void test_vst1_s8_x3(int8_t *a, int8x8x3_t b) {
  vst1_s8_x3(a, b);
}

// CHECK-LABEL: @test_vst1_s8_x4(
// CHECK: [[B:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]])
// CHECK: ret void
void test_vst1_s8_x4(int8_t *a, int8x8x4_t b) {
  vst1_s8_x4(a, b);
}

// CHECK-LABEL: @test_vst1_u16_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v4i16(ptr %a, <4 x i16> [[TMP7]], <4 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1_u16_x2(uint16_t *a, uint16x4x2_t b) {
  vst1_u16_x2(a, b);
}

// CHECK-LABEL: @test_vst1_u16_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v4i16(ptr %a, <4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1_u16_x3(uint16_t *a, uint16x4x3_t b) {
  vst1_u16_x3(a, b);
}

// CHECK-LABEL: @test_vst1_u16_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v4i16(ptr %a, <4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1_u16_x4(uint16_t *a, uint16x4x4_t b) {
  vst1_u16_x4(a, b);
}

// CHECK-LABEL: @test_vst1_u32_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v2i32.p0(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v2i32(ptr %a, <2 x i32> [[TMP7]], <2 x i32> [[TMP8]])
// CHECK: ret void
void test_vst1_u32_x2(uint32_t *a, uint32x2x2_t b) {
  vst1_u32_x2(a, b);
}

// CHECK-LABEL: @test_vst1_u32_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v2i32.p0(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v2i32(ptr %a, <2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]])
// CHECK: ret void
void test_vst1_u32_x3(uint32_t *a, uint32x2x3_t b) {
  vst1_u32_x3(a, b);
}

// CHECK-LABEL: @test_vst1_u32_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <2 x i32>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v2i32.p0(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v2i32(ptr %a, <2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]])
// CHECK: ret void
void test_vst1_u32_x4(uint32_t *a, uint32x2x4_t b) {
  vst1_u32_x4(a, b);
}

// CHECK-LABEL: @test_vst1_u64_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x1x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v1i64(ptr %a, <1 x i64> [[TMP7]], <1 x i64> [[TMP8]])
// CHECK: ret void
void test_vst1_u64_x2(uint64_t *a, uint64x1x2_t b) {
  vst1_u64_x2(a, b);
}

// CHECK-LABEL: @test_vst1_u64_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v1i64(ptr %a, <1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]])
// CHECK: ret void
void test_vst1_u64_x3(uint64_t *a, uint64x1x3_t b) {
  vst1_u64_x3(a, b);
}

// CHECK-LABEL: @test_vst1_u64_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK: [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// CHECK: [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v1i64(ptr %a, <1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]])
// CHECK: ret void
void test_vst1_u64_x4(uint64_t *a, uint64x1x4_t b) {
  vst1_u64_x4(a, b);
}

// CHECK-LABEL: @test_vst1_u8_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [2 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 16, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]])
// CHECK: ret void
void test_vst1_u8_x2(uint8_t *a, uint8x8x2_t b) {
  vst1_u8_x2(a, b);
}

// CHECK-LABEL: @test_vst1_u8_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [3 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 24, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]])
// CHECK: ret void
void test_vst1_u8_x3(uint8_t *a, uint8x8x3_t b) {
  vst1_u8_x3(a, b);
}

// CHECK-LABEL: @test_vst1_u8_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK: [[__S1:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align 8 [[__S1]], ptr align 8 [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v8i8(ptr %a, <8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]])
// CHECK: ret void
void test_vst1_u8_x4(uint8_t *a, uint8x8x4_t b) {
  vst1_u8_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_f16_x2(
// CHECK: [[B:%.*]] = alloca %struct.float16x8x2_t, align [[QALIGN:(16|8)]]
// CHECK: [[__S1:%.*]] = alloca %struct.float16x8x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x half>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x half>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x [[HALF]]>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x [[HALF]]>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v8f16.p0(<8 x half> [[TMP7]], <8 x half> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v8i16(ptr %a, <8 x i16> [[TMP7]], <8 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1q_f16_x2(float16_t *a, float16x8x2_t b) {
  vst1q_f16_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_f16_x3(
// CHECK: [[B:%.*]] = alloca %struct.float16x8x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.float16x8x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x half>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x [[HALF]]>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x [[HALF]]>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x [[HALF]]>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v8f16.p0(<8 x half> [[TMP9]], <8 x half> [[TMP10]], <8 x half> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v8i16(ptr %a, <8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1q_f16_x3(float16_t *a, float16x8x3_t b) {
  vst1q_f16_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_f16_x4(
// CHECK: [[B:%.*]] = alloca %struct.float16x8x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.float16x8x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x half>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <8 x half>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <8 x half> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x [[HALF]]>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x [[HALF]]>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x [[HALF]]>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x [[HALF]]>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v8f16.p0(<8 x half> [[TMP11]], <8 x half> [[TMP12]], <8 x half> [[TMP13]], <8 x half> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v8i16(ptr %a, <8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1q_f16_x4(float16_t *a, float16x8x4_t b) {
  vst1q_f16_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_f32_x2(
// CHECK: [[B:%.*]] = alloca %struct.float32x4x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.float32x4x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x float>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x float>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v4f32.p0(<4 x float> [[TMP7]], <4 x float> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v4f32(ptr %a, <4 x float> [[TMP7]], <4 x float> [[TMP8]])
// CHECK: ret void
void test_vst1q_f32_x2(float32_t *a, float32x4x2_t b) {
  vst1q_f32_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_f32_x3(
// CHECK: [[B:%.*]] = alloca %struct.float32x4x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.float32x4x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x float>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v4f32.p0(<4 x float> [[TMP9]], <4 x float> [[TMP10]], <4 x float> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v4f32(ptr %a, <4 x float> [[TMP9]], <4 x float> [[TMP10]], <4 x float> [[TMP11]])
// CHECK: ret void
void test_vst1q_f32_x3(float32_t *a, float32x4x3_t b) {
  vst1q_f32_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_f32_x4(
// CHECK: [[B:%.*]] = alloca %struct.float32x4x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.float32x4x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x float>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <4 x float>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <4 x float> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x float>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v4f32.p0(<4 x float> [[TMP11]], <4 x float> [[TMP12]], <4 x float> [[TMP13]], <4 x float> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v4f32(ptr %a, <4 x float> [[TMP11]], <4 x float> [[TMP12]], <4 x float> [[TMP13]], <4 x float> [[TMP14]])
// CHECK: ret void
void test_vst1q_f32_x4(float32_t *a, float32x4x4_t b) {
  vst1q_f32_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_p16_x2(
// CHECK: [[B:%.*]] = alloca %struct.poly16x8x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.poly16x8x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v8i16(ptr %a, <8 x i16> [[TMP7]], <8 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1q_p16_x2(poly16_t *a, poly16x8x2_t b) {
  vst1q_p16_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_p16_x3(
// CHECK: [[B:%.*]] = alloca %struct.poly16x8x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.poly16x8x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v8i16(ptr %a, <8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1q_p16_x3(poly16_t *a, poly16x8x3_t b) {
  vst1q_p16_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_p16_x4(
// CHECK: [[B:%.*]] = alloca %struct.poly16x8x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.poly16x8x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v8i16(ptr %a, <8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1q_p16_x4(poly16_t *a, poly16x8x4_t b) {
  vst1q_p16_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_p8_x2(
// CHECK: [[B:%.*]] = alloca %struct.poly8x16x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.poly8x16x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x16x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]])
// CHECK: ret void
void test_vst1q_p8_x2(poly8_t *a, poly8x16x2_t b) {
  vst1q_p8_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_p8_x3(
// CHECK: [[B:%.*]] = alloca %struct.poly8x16x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.poly8x16x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]])
// CHECK: ret void
void test_vst1q_p8_x3(poly8_t *a, poly8x16x3_t b) {
  vst1q_p8_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_p8_x4(
// CHECK: [[B:%.*]] = alloca %struct.poly8x16x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.poly8x16x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]])
// CHECK: ret void
void test_vst1q_p8_x4(poly8_t *a, poly8x16x4_t b) {
  vst1q_p8_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_s16_x2(
// CHECK: [[B:%.*]] = alloca %struct.int16x8x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int16x8x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v8i16(ptr %a, <8 x i16> [[TMP7]], <8 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1q_s16_x2(int16_t *a, int16x8x2_t b) {
  vst1q_s16_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_s16_x3(
// CHECK: [[B:%.*]] = alloca %struct.int16x8x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int16x8x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v8i16(ptr %a, <8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1q_s16_x3(int16_t *a, int16x8x3_t b) {
  vst1q_s16_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_s16_x4(
// CHECK: [[B:%.*]] = alloca %struct.int16x8x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int16x8x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v8i16(ptr %a, <8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1q_s16_x4(int16_t *a, int16x8x4_t b) {
  vst1q_s16_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_s32_x2(
// CHECK: [[B:%.*]] = alloca %struct.int32x4x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int32x4x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v4i32.p0(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v4i32(ptr %a, <4 x i32> [[TMP7]], <4 x i32> [[TMP8]])
// CHECK: ret void
void test_vst1q_s32_x2(int32_t *a, int32x4x2_t b) {
  vst1q_s32_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_s32_x3(
// CHECK: [[B:%.*]] = alloca %struct.int32x4x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int32x4x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v4i32.p0(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v4i32(ptr %a, <4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]])
// CHECK: ret void
void test_vst1q_s32_x3(int32_t *a, int32x4x3_t b) {
  vst1q_s32_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_s32_x4(
// CHECK: [[B:%.*]] = alloca %struct.int32x4x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int32x4x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <4 x i32>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v4i32.p0(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v4i32(ptr %a, <4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]])
// CHECK: ret void
void test_vst1q_s32_x4(int32_t *a, int32x4x4_t b) {
  vst1q_s32_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_s64_x2(
// CHECK: [[B:%.*]] = alloca %struct.int64x2x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int64x2x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v2i64(ptr %a, <2 x i64> [[TMP7]], <2 x i64> [[TMP8]])
// CHECK: ret void
void test_vst1q_s64_x2(int64_t *a, int64x2x2_t b) {
  vst1q_s64_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_s64_x3(
// CHECK: [[B:%.*]] = alloca %struct.int64x2x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int64x2x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v2i64(ptr %a, <2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]])
// CHECK: ret void
void test_vst1q_s64_x3(int64_t *a, int64x2x3_t b) {
  vst1q_s64_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_s64_x4(
// CHECK: [[B:%.*]] = alloca %struct.int64x2x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int64x2x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v2i64(ptr %a, <2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]])
// CHECK: ret void
void test_vst1q_s64_x4(int64_t *a, int64x2x4_t b) {
  vst1q_s64_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_s8_x2(
// CHECK: [[B:%.*]] = alloca %struct.int8x16x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int8x16x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x16x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]])
// CHECK: ret void
void test_vst1q_s8_x2(int8_t *a, int8x16x2_t b) {
  vst1q_s8_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_s8_x3(
// CHECK: [[B:%.*]] = alloca %struct.int8x16x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int8x16x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]])
// CHECK: ret void
void test_vst1q_s8_x3(int8_t *a, int8x16x3_t b) {
  vst1q_s8_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_s8_x4(
// CHECK: [[B:%.*]] = alloca %struct.int8x16x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.int8x16x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]])
// CHECK: ret void
void test_vst1q_s8_x4(int8_t *a, int8x16x4_t b) {
  vst1q_s8_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_u16_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint16x8x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint16x8x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v8i16(ptr %a, <8 x i16> [[TMP7]], <8 x i16> [[TMP8]])
// CHECK: ret void
void test_vst1q_u16_x2(uint16_t *a, uint16x8x2_t b) {
  vst1q_u16_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_u16_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint16x8x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint16x8x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v8i16(ptr %a, <8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]])
// CHECK: ret void
void test_vst1q_u16_x3(uint16_t *a, uint16x8x3_t b) {
  vst1q_u16_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_u16_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint16x8x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint16x8x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v8i16(ptr %a, <8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]])
// CHECK: ret void
void test_vst1q_u16_x4(uint16_t *a, uint16x8x4_t b) {
  vst1q_u16_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_u32_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint32x4x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint32x4x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v4i32.p0(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v4i32(ptr %a, <4 x i32> [[TMP7]], <4 x i32> [[TMP8]])
// CHECK: ret void
void test_vst1q_u32_x2(uint32_t *a, uint32x4x2_t b) {
  vst1q_u32_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_u32_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint32x4x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint32x4x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v4i32.p0(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v4i32(ptr %a, <4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]])
// CHECK: ret void
void test_vst1q_u32_x3(uint32_t *a, uint32x4x3_t b) {
  vst1q_u32_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_u32_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint32x4x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint32x4x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <4 x i32>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v4i32.p0(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v4i32(ptr %a, <4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]])
// CHECK: ret void
void test_vst1q_u32_x4(uint32_t *a, uint32x4x4_t b) {
  vst1q_u32_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_u64_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint64x2x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint64x2x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK-DAG: [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK-DAG: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v2i64(ptr %a, <2 x i64> [[TMP7]], <2 x i64> [[TMP8]])
// CHECK: ret void
void test_vst1q_u64_x2(uint64_t *a, uint64x2x2_t b) {
  vst1q_u64_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_u64_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint64x2x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint64x2x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK-DAG: [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK-DAG: [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v2i64(ptr %a, <2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]])
// CHECK: ret void
void test_vst1q_u64_x3(uint64_t *a, uint64x2x3_t b) {
  vst1q_u64_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_u64_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint64x2x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint64x2x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK: [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK-DAG: [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK-DAG: [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK-DAG: [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK-DAG: [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v2i64(ptr %a, <2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]])
// CHECK: ret void
void test_vst1q_u64_x4(uint64_t *a, uint64x2x4_t b) {
  vst1q_u64_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_u8_x2(
// CHECK: [[B:%.*]] = alloca %struct.uint8x16x2_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint8x16x2_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x16x2_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [4 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 32, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x2.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]])
// CHECK: ret void
void test_vst1q_u8_x2(uint8_t *a, uint8x16x2_t b) {
  vst1q_u8_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_u8_x3(
// CHECK: [[B:%.*]] = alloca %struct.uint8x16x3_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint8x16x3_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [6 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 48, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x3.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]])
// CHECK: ret void
void test_vst1q_u8_x3(uint8_t *a, uint8x16x3_t b) {
  vst1q_u8_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_u8_x4(
// CHECK: [[B:%.*]] = alloca %struct.uint8x16x4_t, align [[QALIGN]]
// CHECK: [[__S1:%.*]] = alloca %struct.uint8x16x4_t, align [[QALIGN]]
// CHECK: [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[B]], i32 0, i32 0
// CHECK-A64: store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK-A32: store [8 x i64] %b.coerce, ptr %coerce.dive, align 8
// CHECK: call void @llvm.memcpy.p0.p0.{{i64|i32}}(ptr align [[QALIGN]] [[__S1]], ptr align [[QALIGN]] [[B]], {{i64|i32}} 64, i1 false)
// CHECK: [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], {{i64|i32}} 0, {{i64|i32}} 0
// CHECK: [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align [[QALIGN]]
// CHECK: [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], {{i64|i32}} 0, {{i64|i32}} 1
// CHECK: [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align [[QALIGN]]
// CHECK: [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], {{i64|i32}} 0, {{i64|i32}} 2
// CHECK: [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align [[QALIGN]]
// CHECK: [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK: [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], {{i64|i32}} 0, {{i64|i32}} 3
// CHECK: [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align [[QALIGN]]
// CHECK-A64: call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// CHECK-A32: call void @llvm.arm.neon.vst1x4.p0.v16i8(ptr %a, <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]])
// CHECK: ret void
void test_vst1q_u8_x4(uint8_t *a, uint8x16x4_t b) {
  vst1q_u8_x4(a, b);
}
