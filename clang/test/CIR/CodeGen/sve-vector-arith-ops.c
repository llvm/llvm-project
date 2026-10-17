// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN: -disable-O0-optnone -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN: -disable-O0-optnone \
// RUN:  -fclangir -emit-llvm -o - %s | opt -S -passes=sroa \
// RUN:  | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN: -disable-O0-optnone \
// RUN:  -emit-llvm -o - %s | opt -S -passes=sroa \
// RUN:  | FileCheck --check-prefixes=LLVM,OGCG %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

// ADDITION

// CIR-LABEL: cir.func {{.*}} @add_i8(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_i8(svint8_t a, svint8_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_i16(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @add_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[ADD]]
//
svint16_t add_i16(svint16_t a, svint16_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_i32(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @add_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[ADD]]
//
svint32_t add_i32(svint32_t a, svint32_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_i64(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @add_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[ADD]]
//
svint64_t add_i64(svint64_t a, svint64_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_u8(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @add_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svuint8_t add_u8(svuint8_t a, svuint8_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_u16(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @add_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[ADD]]
//
svuint16_t add_u16(svuint16_t a, svuint16_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_u32(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @add_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[ADD]]
//
svuint32_t add_u32(svuint32_t a, svuint32_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_u64(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @add_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[ADD]]
//
svuint64_t add_u64(svuint64_t a, svuint64_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_f16(
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @add_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[ADD]]
//
svfloat16_t add_f16(svfloat16_t a, svfloat16_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_f32(
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @add_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[ADD]]
//
svfloat32_t add_f32(svfloat32_t a, svfloat32_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_f64(
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_f64(svfloat64_t a, svfloat64_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_i8(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_inplace_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_inplace_i8(svint8_t a, svint8_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_i16(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @add_inplace_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[ADD]]
//
svint16_t add_inplace_i16(svint16_t a, svint16_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_i32(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @add_inplace_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[ADD]]
//
svint32_t add_inplace_i32(svint32_t a, svint32_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_i64(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @add_inplace_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[ADD]]
//
svint64_t add_inplace_i64(svint64_t a, svint64_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_u8(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @add_inplace_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svuint8_t add_inplace_u8(svuint8_t a, svuint8_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_u16(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @add_inplace_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[ADD]]
//
svuint16_t add_inplace_u16(svuint16_t a, svuint16_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_u32(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @add_inplace_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[ADD]]
//
svuint32_t add_inplace_u32(svuint32_t a, svuint32_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_u64(
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @add_inplace_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[ADD]]
//
svuint64_t add_inplace_u64(svuint64_t a, svuint64_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_f16(
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @add_inplace_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[ADD]]
//
svfloat16_t add_inplace_f16(svfloat16_t a, svfloat16_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_f32(
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @add_inplace_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[ADD]]
//
svfloat32_t add_inplace_f32(svfloat32_t a, svfloat32_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_inplace_f64(
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_inplace_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_inplace_f64(svfloat64_t a, svfloat64_t b) {
  return a += b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_i8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_scalar_i8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_scalar_i8(svint8_t a, int8_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_i16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s16i, !cir.vector<[8] x !s16i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @add_scalar_i16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[ADD]]
//
svint16_t add_scalar_i16(svint16_t a, int16_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_i32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s32i, !cir.vector<[4] x !s32i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @add_scalar_i32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[ADD]]
//
svint32_t add_scalar_i32(svint32_t a, int32_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_i64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s64i, !cir.vector<[2] x !s64i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @add_scalar_i64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[ADD]]
//
svint64_t add_scalar_i64(svint64_t a, int64_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_u8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u8i, !cir.vector<[16] x !u8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @add_scalar_u8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svuint8_t add_scalar_u8(svuint8_t a, uint8_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_u16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u16i, !cir.vector<[8] x !u16i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @add_scalar_u16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[ADD]]
//
svuint16_t add_scalar_u16(svuint16_t a, uint16_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_u32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u32i, !cir.vector<[4] x !u32i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @add_scalar_u32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[ADD]]
//
svuint32_t add_scalar_u32(svuint32_t a, uint32_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_u64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u64i, !cir.vector<[2] x !u64i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @add_scalar_u64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[ADD]]
//
svuint64_t add_scalar_u64(svuint64_t a, uint64_t b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_f16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.f16, !cir.vector<[8] x !cir.f16>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @add_scalar_f16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x half> poison, half [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x half> [[SPLAT_SPLATINSERT]], <vscale x 8 x half> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 8 x half> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[ADD]]
//
svfloat16_t add_scalar_f16(svfloat16_t a, __fp16 b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_f32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.float, !cir.vector<[4] x !cir.float>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @add_scalar_f32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x float> poison, float [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[SPLAT_SPLATINSERT]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 4 x float> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[ADD]]
//
svfloat32_t add_scalar_f32(svfloat32_t a, float b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_scalar_f64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_scalar_f64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x double> poison, double [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x double> [[SPLAT_SPLATINSERT]], <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_scalar_f64(svfloat64_t a, double b) {
  return a + b;
}

// CIR-LABEL: cir.func {{.*}} @add_i8_i_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s8i
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_i8_i_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_i8_i_lit(svint8_t a) {
  return a + 0;
}

// CIR-LABEL: cir.func {{.*}} @add_i8_il_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s8i
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_i8_il_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_i8_il_lit(svint8_t a) {
  return a + 0l;
}

// CIR-LABEL: cir.func {{.*}} @add_i8_ill_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s8i
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_i8_ill_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_i8_ill_lit(svint8_t a) {
  return a + 0ll;
}

// CIR-LABEL: cir.func {{.*}} @add_i8_u_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s8i
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_i8_u_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_i8_u_lit(svint8_t a) {
  return a + 0u;
}

// CIR-LABEL: cir.func {{.*}} @add_i8_ul_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s8i
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_i8_ul_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_i8_ul_lit(svint8_t a) {
  return a + 0ul;
}

// CIR-LABEL: cir.func {{.*}} @add_i8_ull_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s8i
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @add_i8_ull_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = add <vscale x 16 x i8> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 16 x i8> [[ADD]]
//
svint8_t add_i8_ull_lit(svint8_t a) {
  return a + 0ull;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_i_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s32i
// CIR: {{%.*}} = cir.cast int_to_float {{%.*}} : !s32i -> !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_i_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_f64_i_lit(svfloat64_t a) {
  return a + 0;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_il_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s64i
// CIR: {{%.*}} = cir.cast int_to_float {{%.*}} : !s64i -> !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_il_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_f64_il_lit(svfloat64_t a) {
  return a + 0l;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_ill_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !s64i
// CIR: {{%.*}} = cir.cast int_to_float {{%.*}} : !s64i -> !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_ill_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_f64_ill_lit(svfloat64_t a) {
  return a + 0ll;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_u_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !u32i
// CIR: {{%.*}} = cir.cast int_to_float {{%.*}} : !u32i -> !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_u_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_f64_u_lit(svfloat64_t a) {
  return a + 0u;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_ul_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !u64i
// CIR: {{%.*}} = cir.cast int_to_float {{%.*}} : !u64i -> !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_ul_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]

svfloat64_t add_f64_ul_lit(svfloat64_t a) {
  return a + 0ul;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_ull_lit(
// CIR: {{%.*}} = cir.const #cir.int<0> : !u64i
// CIR: {{%.*}} = cir.cast int_to_float {{%.*}} : !u64i -> !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_ull_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]

svfloat64_t add_f64_ull_lit(svfloat64_t a) {
  return a + 0ull;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_f_lit(
// CIR: {{%.*}} = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR: {{%.*}} = cir.cast floating {{%.*}} : !cir.float -> !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_f_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_f64_f_lit(svfloat64_t a) {
  return a + 0.f;
}

// CIR-LABEL: cir.func {{.*}} @add_f64_d_lit(
// CIR: {{%.*}} = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fadd {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @add_f64_d_lit(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[ADD:%.*]] = fadd <vscale x 2 x double> [[A:%.*]], zeroinitializer
// LLVM-NEXT:    ret <vscale x 2 x double> [[ADD]]
//
svfloat64_t add_f64_d_lit(svfloat64_t a) {
  return a + 0.;
}

// SUBTRACTION

// CIR-LABEL: cir.func {{.*}} @sub_i8(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @sub_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svint8_t sub_i8(svint8_t a, svint8_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_i16(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @sub_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svint16_t sub_i16(svint16_t a, svint16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_i32(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @sub_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svint32_t sub_i32(svint32_t a, svint32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_i64(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @sub_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svint64_t sub_i64(svint64_t a, svint64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_u8(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @sub_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svuint8_t sub_u8(svuint8_t a, svuint8_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_u16(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @sub_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svuint16_t sub_u16(svuint16_t a, svuint16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_u32(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @sub_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svuint32_t sub_u32(svuint32_t a, svuint32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_u64(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @sub_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svuint64_t sub_u64(svuint64_t a, svuint64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_f16(
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @sub_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[SUB]]
//
svfloat16_t sub_f16(svfloat16_t a, svfloat16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_f32(
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @sub_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[SUB]]
//
svfloat32_t sub_f32(svfloat32_t a, svfloat32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_f64(
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @sub_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[SUB]]
//
svfloat64_t sub_f64(svfloat64_t a, svfloat64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_i8(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @sub_inplace_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svint8_t sub_inplace_i8(svint8_t a, svint8_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_i16(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @sub_inplace_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svint16_t sub_inplace_i16(svint16_t a, svint16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_i32(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @sub_inplace_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svint32_t sub_inplace_i32(svint32_t a, svint32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_i64(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @sub_inplace_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svint64_t sub_inplace_i64(svint64_t a, svint64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_u8(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @sub_inplace_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svuint8_t sub_inplace_u8(svuint8_t a, svuint8_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_u16(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @sub_inplace_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svuint16_t sub_inplace_u16(svuint16_t a, svuint16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_u32(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @sub_inplace_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svuint32_t sub_inplace_u32(svuint32_t a, svuint32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_u64(
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @sub_inplace_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svuint64_t sub_inplace_u64(svuint64_t a, svuint64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_f16(
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @sub_inplace_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[SUB]]
//
svfloat16_t sub_inplace_f16(svfloat16_t a, svfloat16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_f32(
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @sub_inplace_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[SUB]]
//
svfloat32_t sub_inplace_f32(svfloat32_t a, svfloat32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_inplace_f64(
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @sub_inplace_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[SUB]]
//
svfloat64_t sub_inplace_f64(svfloat64_t a, svfloat64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_i8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @sub_scalar_i8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svint8_t sub_scalar_i8(svint8_t a, int8_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_i16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s16i, !cir.vector<[8] x !s16i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @sub_scalar_i16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svint16_t sub_scalar_i16(svint16_t a, int16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_i32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s32i, !cir.vector<[4] x !s32i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @sub_scalar_i32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svint32_t sub_scalar_i32(svint32_t a, int32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_i64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s64i, !cir.vector<[2] x !s64i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @sub_scalar_i64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svint64_t sub_scalar_i64(svint64_t a, int64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_u8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u8i, !cir.vector<[16] x !u8i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @sub_scalar_u8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svuint8_t sub_scalar_u8(svuint8_t a, uint8_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_u16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u16i, !cir.vector<[8] x !u16i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @sub_scalar_u16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svuint16_t sub_scalar_u16(svuint16_t a, uint16_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_u32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u32i, !cir.vector<[4] x !u32i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @sub_scalar_u32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svuint32_t sub_scalar_u32(svuint32_t a, uint32_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_u64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u64i, !cir.vector<[2] x !u64i>
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @sub_scalar_u64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svuint64_t sub_scalar_u64(svuint64_t a, uint64_t b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_f16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.f16, !cir.vector<[8] x !cir.f16>
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @sub_scalar_f16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x half> poison, half [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x half> [[SPLAT_SPLATINSERT]], <vscale x 8 x half> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 8 x half> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[SUB]]
//
svfloat16_t sub_scalar_f16(svfloat16_t a, __fp16 b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_f32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.float, !cir.vector<[4] x !cir.float>
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @sub_scalar_f32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x float> poison, float [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[SPLAT_SPLATINSERT]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 4 x float> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[SUB]]
//
svfloat32_t sub_scalar_f32(svfloat32_t a, float b) {
  return a - b;
}

// CIR-LABEL: cir.func {{.*}} @sub_scalar_f64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fsub {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @sub_scalar_f64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x double> poison, double [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x double> [[SPLAT_SPLATINSERT]], <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[SUB:%.*]] = fsub <vscale x 2 x double> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[SUB]]
//
svfloat64_t sub_scalar_f64(svfloat64_t a, double b) {
  return a - b;
}

// MULTIPLICATION

// CIR-LABEL: cir.func {{.*}} @mul_i8(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @mul_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[MUL]]
//
svint8_t mul_i8(svint8_t a, svint8_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_i16(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @mul_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[MUL]]
//
svint16_t mul_i16(svint16_t a, svint16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_i32(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @mul_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[MUL]]
//
svint32_t mul_i32(svint32_t a, svint32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_i64(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @mul_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[MUL]]
//
svint64_t mul_i64(svint64_t a, svint64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_u8(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @mul_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[MUL]]
//
svuint8_t mul_u8(svuint8_t a, svuint8_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_u16(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @mul_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[MUL]]
//
svuint16_t mul_u16(svuint16_t a, svuint16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_u32(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @mul_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[MUL]]
//
svuint32_t mul_u32(svuint32_t a, svuint32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_u64(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @mul_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[MUL]]
//
svuint64_t mul_u64(svuint64_t a, svuint64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_f16(
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @mul_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[MUL]]
//
svfloat16_t mul_f16(svfloat16_t a, svfloat16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_f32(
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @mul_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[MUL]]
//
svfloat32_t mul_f32(svfloat32_t a, svfloat32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_f64(
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @mul_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[MUL]]
//
svfloat64_t mul_f64(svfloat64_t a, svfloat64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_i8(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @mul_inplace_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[MUL]]
//
svint8_t mul_inplace_i8(svint8_t a, svint8_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_i16(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @mul_inplace_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[MUL]]
//
svint16_t mul_inplace_i16(svint16_t a, svint16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_i32(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @mul_inplace_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[MUL]]
//
svint32_t mul_inplace_i32(svint32_t a, svint32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_i64(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @mul_inplace_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[MUL]]
//
svint64_t mul_inplace_i64(svint64_t a, svint64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_u8(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @mul_inplace_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[MUL]]
//
svuint8_t mul_inplace_u8(svuint8_t a, svuint8_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_u16(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @mul_inplace_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[MUL]]
//
svuint16_t mul_inplace_u16(svuint16_t a, svuint16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_u32(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @mul_inplace_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[MUL]]
//
svuint32_t mul_inplace_u32(svuint32_t a, svuint32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_u64(
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @mul_inplace_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[MUL]]
//
svuint64_t mul_inplace_u64(svuint64_t a, svuint64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_f16(
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @mul_inplace_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[MUL]]
//
svfloat16_t mul_inplace_f16(svfloat16_t a, svfloat16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_f32(
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @mul_inplace_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[MUL]]
//
svfloat32_t mul_inplace_f32(svfloat32_t a, svfloat32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_inplace_f64(
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @mul_inplace_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[MUL]]
//
svfloat64_t mul_inplace_f64(svfloat64_t a, svfloat64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_i8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @mul_scalar_i8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[MUL]]
//
svint8_t mul_scalar_i8(svint8_t a, int8_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_i16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s16i, !cir.vector<[8] x !s16i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @mul_scalar_i16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[MUL]]

svint16_t mul_scalar_i16(svint16_t a, int16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_i32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s32i, !cir.vector<[4] x !s32i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @mul_scalar_i32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[MUL]]
//
svint32_t mul_scalar_i32(svint32_t a, int32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_i64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s64i, !cir.vector<[2] x !s64i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @mul_scalar_i64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[MUL]]
//
svint64_t mul_scalar_i64(svint64_t a, int64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_u8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u8i, !cir.vector<[16] x !u8i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @mul_scalar_u8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[MUL]]
//
svuint8_t mul_scalar_u8(svuint8_t a, uint8_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_u16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u16i, !cir.vector<[8] x !u16i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @mul_scalar_u16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[MUL]]
//
svuint16_t mul_scalar_u16(svuint16_t a, uint16_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_u32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u32i, !cir.vector<[4] x !u32i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @mul_scalar_u32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[MUL]]
//
svuint32_t mul_scalar_u32(svuint32_t a, uint32_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_u64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u64i, !cir.vector<[2] x !u64i>
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @mul_scalar_u64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = mul <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[MUL]]
//
svuint64_t mul_scalar_u64(svuint64_t a, uint64_t b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_f16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.f16, !cir.vector<[8] x !cir.f16>
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @mul_scalar_f16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x half> poison, half [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x half> [[SPLAT_SPLATINSERT]], <vscale x 8 x half> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 8 x half> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[MUL]]
//
svfloat16_t mul_scalar_f16(svfloat16_t a, __fp16 b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_f32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.float, !cir.vector<[4] x !cir.float>
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @mul_scalar_f32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x float> poison, float [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[SPLAT_SPLATINSERT]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 4 x float> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[MUL]]
//
svfloat32_t mul_scalar_f32(svfloat32_t a, float b) {
  return a * b;
}

// CIR-LABEL: cir.func {{.*}} @mul_scalar_f64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fmul {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @mul_scalar_f64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x double> poison, double [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x double> [[SPLAT_SPLATINSERT]], <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[MUL:%.*]] = fmul <vscale x 2 x double> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[MUL]]
//
svfloat64_t mul_scalar_f64(svfloat64_t a, double b) {
  return a * b;
}

// DIVISION

// CIR-LABEL: cir.func {{.*}} @div_i8(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @div_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[DIV]]
//
svint8_t div_i8(svint8_t a, svint8_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_i16(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @div_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[DIV]]
//
svint16_t div_i16(svint16_t a, svint16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_i32(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @div_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[DIV]]
//
svint32_t div_i32(svint32_t a, svint32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_i64(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @div_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[DIV]]
//
svint64_t div_i64(svint64_t a, svint64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_u8(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @div_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[DIV]]
//
svuint8_t div_u8(svuint8_t a, svuint8_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_u16(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @div_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[DIV]]
//
svuint16_t div_u16(svuint16_t a, svuint16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_u32(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @div_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[DIV]]
//
svuint32_t div_u32(svuint32_t a, svuint32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_u64(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @div_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[DIV]]
//
svuint64_t div_u64(svuint64_t a, svuint64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_f16(
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @div_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[DIV]]
//
svfloat16_t div_f16(svfloat16_t a, svfloat16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_f32(
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @div_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[DIV]]
//
svfloat32_t div_f32(svfloat32_t a, svfloat32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_f64(
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @div_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[DIV]]
//
svfloat64_t div_f64(svfloat64_t a, svfloat64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_i8(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @div_inplace_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[DIV]]
//
svint8_t div_inplace_i8(svint8_t a, svint8_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_i16(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @div_inplace_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[DIV]]
//
svint16_t div_inplace_i16(svint16_t a, svint16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_i32(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @div_inplace_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[DIV]]
//
svint32_t div_inplace_i32(svint32_t a, svint32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_i64(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @div_inplace_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[DIV]]
//
svint64_t div_inplace_i64(svint64_t a, svint64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_u8(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @div_inplace_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[DIV]]
//
svuint8_t div_inplace_u8(svuint8_t a, svuint8_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_u16(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @div_inplace_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[DIV]]
//
svuint16_t div_inplace_u16(svuint16_t a, svuint16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_u32(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @div_inplace_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[DIV]]
//
svuint32_t div_inplace_u32(svuint32_t a, svuint32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_u64(
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @div_inplace_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[DIV]]
//
svuint64_t div_inplace_u64(svuint64_t a, svuint64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_f16(
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @div_inplace_f16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 8 x half> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[DIV]]
//
svfloat16_t div_inplace_f16(svfloat16_t a, svfloat16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_f32(
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @div_inplace_f32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 4 x float> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[DIV]]
//
svfloat32_t div_inplace_f32(svfloat32_t a, svfloat32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_inplace_f64(
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @div_inplace_f64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 2 x double> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[DIV]]
//
svfloat64_t div_inplace_f64(svfloat64_t a, svfloat64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_i8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @div_scalar_i8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[DIV]]
//
svint8_t div_scalar_i8(svint8_t a, int8_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_i16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s16i, !cir.vector<[8] x !s16i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @div_scalar_i16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[DIV]]
//
svint16_t div_scalar_i16(svint16_t a, int16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_i32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s32i, !cir.vector<[4] x !s32i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @div_scalar_i32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[DIV]]
//
svint32_t div_scalar_i32(svint32_t a, int32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_i64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s64i, !cir.vector<[2] x !s64i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @div_scalar_i64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = sdiv <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[DIV]]
//
svint64_t div_scalar_i64(svint64_t a, int64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_u8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u8i, !cir.vector<[16] x !u8i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @div_scalar_u8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[DIV]]
//
svuint8_t div_scalar_u8(svuint8_t a, uint8_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_u16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u16i, !cir.vector<[8] x !u16i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @div_scalar_u16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[DIV]]
//
svuint16_t div_scalar_u16(svuint16_t a, uint16_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_u32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u32i, !cir.vector<[4] x !u32i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @div_scalar_u32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[DIV]]
//
svuint32_t div_scalar_u32(svuint32_t a, uint32_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_u64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u64i, !cir.vector<[2] x !u64i>
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @div_scalar_u64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = udiv <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[DIV]]
//
svuint64_t div_scalar_u64(svuint64_t a, uint64_t b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_f16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.f16, !cir.vector<[8] x !cir.f16>
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[8] x !cir.f16>

// LLVM-LABEL: @div_scalar_f16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x half> poison, half [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x half> [[SPLAT_SPLATINSERT]], <vscale x 8 x half> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 8 x half> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x half> [[DIV]]
//
svfloat16_t div_scalar_f16(svfloat16_t a, __fp16 b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_f32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.float, !cir.vector<[4] x !cir.float>
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[4] x !cir.float>

// LLVM-LABEL: @div_scalar_f32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x float> poison, float [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[SPLAT_SPLATINSERT]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 4 x float> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x float> [[DIV]]
//
svfloat32_t div_scalar_f32(svfloat32_t a, float b) {
  return a / b;
}

// CIR-LABEL: cir.func {{.*}} @div_scalar_f64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !cir.double, !cir.vector<[2] x !cir.double>
// CIR: {{%.*}} = cir.fdiv {{%.*}}, {{%.*}} : !cir.vector<[2] x !cir.double>

// LLVM-LABEL: @div_scalar_f64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x double> poison, double [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x double> [[SPLAT_SPLATINSERT]], <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[DIV:%.*]] = fdiv <vscale x 2 x double> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x double> [[DIV]]
//
svfloat64_t div_scalar_f64(svfloat64_t a, double b) {
  return a / b;
}

// REMAINDER

// CIR-LABEL: cir.func {{.*}} @rem_i8(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @rem_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[REM]]
//
svint8_t rem_i8(svint8_t a, svint8_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_i16(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @rem_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[REM]]
//
svint16_t rem_i16(svint16_t a, svint16_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_i32(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @rem_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[REM]]
//
svint32_t rem_i32(svint32_t a, svint32_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_i64(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @rem_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[REM]]
//
svint64_t rem_i64(svint64_t a, svint64_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_u8(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @rem_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[REM]]
//
svuint8_t rem_u8(svuint8_t a, svuint8_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_u16(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @rem_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[REM]]
//
svuint16_t rem_u16(svuint16_t a, svuint16_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_u32(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @rem_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[REM]]
//
svuint32_t rem_u32(svuint32_t a, svuint32_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_u64(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @rem_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[REM]]
//
svuint64_t rem_u64(svuint64_t a, svuint64_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_i8(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @rem_inplace_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[REM]]
//
svint8_t rem_inplace_i8(svint8_t a, svint8_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_i16(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @rem_inplace_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[REM]]
//
svint16_t rem_inplace_i16(svint16_t a, svint16_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_i32(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @rem_inplace_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[REM]]
//
svint32_t rem_inplace_i32(svint32_t a, svint32_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_i64(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @rem_inplace_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[REM]]
//
svint64_t rem_inplace_i64(svint64_t a, svint64_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_u8(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @rem_inplace_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 16 x i8> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[REM]]
//
svuint8_t rem_inplace_u8(svuint8_t a, svuint8_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_u16(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @rem_inplace_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 8 x i16> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[REM]]
//
svuint16_t rem_inplace_u16(svuint16_t a, svuint16_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_u32(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @rem_inplace_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 4 x i32> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[REM]]
//
svuint32_t rem_inplace_u32(svuint32_t a, svuint32_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_inplace_u64(
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @rem_inplace_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 2 x i64> [[A:%.*]], [[B:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[REM]]
//
svuint64_t rem_inplace_u64(svuint64_t a, svuint64_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_i8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s8i, !cir.vector<[16] x !s8i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @rem_scalar_i8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[REM]]
//
svint8_t rem_scalar_i8(svint8_t a, int8_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_i16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s16i, !cir.vector<[8] x !s16i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @rem_scalar_i16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[REM]]
//
svint16_t rem_scalar_i16(svint16_t a, int16_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_i32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s32i, !cir.vector<[4] x !s32i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @rem_scalar_i32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[REM]]
//
svint32_t rem_scalar_i32(svint32_t a, int32_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_i64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !s64i, !cir.vector<[2] x !s64i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @rem_scalar_i64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = srem <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[REM]]
//
svint64_t rem_scalar_i64(svint64_t a, int64_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_u8(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u8i, !cir.vector<[16] x !u8i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @rem_scalar_u8(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 16 x i8> poison, i8 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 16 x i8> [[SPLAT_SPLATINSERT]], <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 16 x i8> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[REM]]
//
svuint8_t rem_scalar_u8(svuint8_t a, uint8_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_u16(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u16i, !cir.vector<[8] x !u16i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @rem_scalar_u16(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i16> poison, i16 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 8 x i16> [[SPLAT_SPLATINSERT]], <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 8 x i16> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[REM]]
//
svuint16_t rem_scalar_u16(svuint16_t a, uint16_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_u32(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u32i, !cir.vector<[4] x !u32i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @rem_scalar_u32(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 4 x i32> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[REM]]
//
svuint32_t rem_scalar_u32(svuint32_t a, uint32_t b) {
  return a % b;
}

// CIR-LABEL: cir.func {{.*}} @rem_scalar_u64(
// CIR: {{%.*}} = cir.vec.splat {{%.*}} : !u64i, !cir.vector<[2] x !u64i>
// CIR: {{%.*}} = cir.rem {{%.*}}, {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @rem_scalar_u64(
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[SPLAT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[B:%.*]], i64 0
// LLVM-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[SPLAT_SPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
// LLVM-NEXT:    [[REM:%.*]] = urem <vscale x 2 x i64> [[A:%.*]], [[SPLAT_SPLAT]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[REM]]
//
svuint64_t rem_scalar_u64(svuint64_t a, uint64_t b) {
  return a % b;
}

// UNARY PROMOTION

// CIR-LABEL: cir.func {{.*}} @prom_i8(
// CIR: cir.return {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @prom_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 16 x i8> [[A:%.*]]
//
svint8_t prom_i8(svint8_t a) {
  return +a;
}

// CIR-LABEL: cir.func {{.*}} @prom_i16(
// CIR: cir.return {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @prom_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 8 x i16> [[A:%.*]]
//
svint16_t prom_i16(svint16_t a) {
  return +a;
}

// CIR-LABEL: cir.func {{.*}} @prom_i32(
// CIR: cir.return {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @prom_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 4 x i32> [[A:%.*]]
//
svint32_t prom_i32(svint32_t a) {
  return +a;
}

// CIR-LABEL: cir.func {{.*}} @prom_i64(
// CIR: cir.return {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @prom_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 2 x i64> [[A:%.*]]
//
svint64_t prom_i64(svint64_t a) {
  return +a;
}

// CIR-LABEL: cir.func {{.*}} @prom_u8(
// CIR: cir.return {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @prom_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 16 x i8> [[A:%.*]]
//
svuint8_t prom_u8(svuint8_t a) {
  return +a;
}

// CIR-LABEL: cir.func {{.*}} @prom_u16(
// CIR: cir.return {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @prom_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 8 x i16> [[A:%.*]]
//
svuint16_t prom_u16(svuint16_t a) {
  return +a;
}

// CIR-LABEL: cir.func {{.*}} @prom_u32(
// CIR: cir.return {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @prom_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 4 x i32> [[A:%.*]]
//
svuint32_t prom_u32(svuint32_t a) {
  return +a;
}

// CIR-LABEL: cir.func {{.*}} @prom_u64(
// CIR: cir.return {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @prom_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    ret <vscale x 2 x i64> [[A:%.*]]
//
svuint64_t prom_u64(svuint64_t a) {
  return +a;
}

// UNARY NEGATION

// CIR-LABEL: cir.func {{.*}} @neg_i8(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[16] x !s8i>

// LLVM-LABEL: @neg_i8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svint8_t neg_i8(svint8_t a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}} @neg_i16(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[8] x !s16i>

// LLVM-LABEL: @neg_i16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svint16_t neg_i16(svint16_t a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}} @neg_i32(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[4] x !s32i>

// LLVM-LABEL: @neg_i32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svint32_t neg_i32(svint32_t a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}} @neg_i64(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[2] x !s64i>

// LLVM-LABEL: @neg_i64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svint64_t neg_i64(svint64_t a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}} @neg_u8(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[16] x !u8i>

// LLVM-LABEL: @neg_u8(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 16 x i8> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 16 x i8> [[SUB]]
//
svuint8_t neg_u8(svuint8_t a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}} @neg_u16(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[8] x !u16i>

// LLVM-LABEL: @neg_u16(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 8 x i16> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 8 x i16> [[SUB]]
//
svuint16_t neg_u16(svuint16_t a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}} @neg_u32(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[4] x !u32i>

// LLVM-LABEL: @neg_u32(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 4 x i32> [[SUB]]
//
svuint32_t neg_u32(svuint32_t a) {
  return -a;
}

// CIR-LABEL: cir.func {{.*}} @neg_u64(
// CIR: {{%.*}} = cir.minus {{%.*}} : !cir.vector<[2] x !u64i>

// LLVM-LABEL: @neg_u64(
// OGCG-NEXT:   entry:
// LLVM-NEXT:    [[SUB:%.*]] = sub <vscale x 2 x i64> zeroinitializer, [[A:%.*]]
// LLVM-NEXT:    ret <vscale x 2 x i64> [[SUB]]
//
svuint64_t neg_u64(svuint64_t a) {
  return -a;
}
