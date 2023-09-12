// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

#ifdef SME_OVERLOADED_FORMS
#define SME_ACLE_FUNC(A1,A2_UNUSED,A3) A1##A3
#else
#define SME_ACLE_FUNC(A1,A2,A3) A1##A2##A3
#endif

// CHECK-C-LABEL: @test_svwrite_hor_za8_s8(
// CHECK-CXX-LABEL: @_Z23test_svwrite_hor_za8_s8ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za8_s8(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za8, _s8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za8_s8_1(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za8_s8_1ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv16i8(i32 0, i32 [[TILESLICE]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za8_s8_1(uint32_t slice_base, svbool_t pg, svint8_t zn) {
   uint32_t slice = slice_base + 15;
  SME_ACLE_FUNC(svwrite_hor_za8, _s8, _m)(0, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_s16(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za16_s16ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_s16(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za16, _s16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_s16_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za16_s16_1ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8i16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_s16_1(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_hor_za16, _s16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za32_s32(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za32_s32ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za32_s32(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za32, _s32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za32_s32_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za32_s32_1ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv4i32(i32 3, i32 [[TILESLICE]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za32_s32_1(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  uint32_t slice = slice_base + 3;
  SME_ACLE_FUNC(svwrite_hor_za32, _s32, _m)(3, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za64_s64(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za64_s64ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za64_s64(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za64, _s64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za64_s64_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za64_s64_1ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv2i64(i32 7, i32 [[TILESLICE]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za64_s64_1(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  uint32_t slice = slice_base + 1;
  SME_ACLE_FUNC(svwrite_hor_za64, _s64, _m)(7, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za8_u8(
// CHECK-CXX-LABEL: @_Z23test_svwrite_hor_za8_u8ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za8_u8(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za8, _u8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za8_u8_1(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za8_u8_1ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv16i8(i32 0, i32 [[TILESLICE]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za8_u8_1(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  uint32_t slice = slice_base + 15;
  SME_ACLE_FUNC(svwrite_hor_za8, _u8, _m)(0, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_u16(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za16_u16ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_u16(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za16, _u16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_u16_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za16_u16_1ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8i16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_u16_1(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_hor_za16, _u16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za32_u32(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za32_u32ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za32_u32(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za32, _u32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za32_u32_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za32_u32_1ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv4i32(i32 3, i32 [[TILESLICE]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za32_u32_1(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  uint32_t slice = slice_base + 3;
  SME_ACLE_FUNC(svwrite_hor_za32, _u32, _m)(3, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za64_u64(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za64_u64ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za64_u64(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za64, _u64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za64_u64_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za64_u64_1ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv2i64(i32 7, i32 [[TILESLICE]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za64_u64_1(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  uint32_t slice = slice_base + 1;
  SME_ACLE_FUNC(svwrite_hor_za64, _u64, _m)(7, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_f16(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za16_f16ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8f16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_f16(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za16, _f16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_f16_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za16_f16_1ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8f16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_f16_1(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_hor_za16, _f16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_bf16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za16_bf16ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8bf16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_bf16(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za16, _bf16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za16_bf16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za16_bf16_1ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv8bf16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za16_bf16_1(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
   uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_hor_za16, _bf16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za32_f32(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za32_f32ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv4f32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za32_f32(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za32, _f32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za32_f32_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za32_f32_1ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv4f32(i32 3, i32 [[TILESLICE]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za32_f32_1(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  uint32_t slice = slice_base + 3;
  SME_ACLE_FUNC(svwrite_hor_za32, _f32, _m)(3, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za64_f64(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za64_f64ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv2f64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za64_f64(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za64, _f64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za64_f64_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za64_f64_1ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.horiz.nxv2f64(i32 7, i32 [[TILESLICE]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za64_f64_1(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  uint32_t slice = slice_base + 1;
  SME_ACLE_FUNC(svwrite_hor_za64, _f64, _m)(7, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s8(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za128_s8ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s8(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s8_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za128_s8_1ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv16i8(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s8_1(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s8, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_s16ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s16(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_s16_1ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8i16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s16_1(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s32(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_s32ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s32(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s32_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_s32_1ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv4i32(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s32_1(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s32, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s64(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_s64ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s64(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_s64_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_s64_1ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv2i64(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_s64_1(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _s64, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u8(
// CHECK-CXX-LABEL: @_Z25test_svwrite_hor_za128_u8ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u8(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u8_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za128_u8_1ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv16i8(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u8_1(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u8, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_u16ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u16(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_u16_1ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8i16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u16_1(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u32(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_u32ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u32(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u32_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_u32_1ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv4i32(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u32_1(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u32, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u64(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_u64ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u64(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_u64_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_u64_1ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv2i64(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_u64_1(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _u64, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_f16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_f16ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8f16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_f16(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _f16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_f16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_f16_1ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8f16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_f16_1(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _f16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_bf16(
// CHECK-CXX-LABEL: @_Z27test_svwrite_hor_za128_bf16ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8bf16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_bf16(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _bf16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_bf16_1(
// CHECK-CXX-LABEL: @_Z29test_svwrite_hor_za128_bf16_1ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv8bf16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_bf16_1(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _bf16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_f32(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_f32ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv4f32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_f32(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _f32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_f32_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_f32_1ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv4f32(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_f32_1(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _f32, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_f64(
// CHECK-CXX-LABEL: @_Z26test_svwrite_hor_za128_f64ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv2f64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_f64(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _f64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_hor_za128_f64_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_hor_za128_f64_1ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.horiz.nxv2f64(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_hor_za128_f64_1(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  SME_ACLE_FUNC(svwrite_hor_za128, _f64, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za8_s8(
// CHECK-CXX-LABEL: @_Z23test_svwrite_ver_za8_s8ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za8_s8(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za8, _s8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za8_s8_1(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za8_s8_1ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv16i8(i32 0, i32 [[TILESLICE]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za8_s8_1(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  uint32_t slice = slice_base + 15;
  SME_ACLE_FUNC(svwrite_ver_za8, _s8, _m)(0, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_s16(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za16_s16ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_s16(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za16, _s16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_s16_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za16_s16_1ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8i16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_s16_1(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_ver_za16, _s16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za32_s32(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za32_s32ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za32_s32(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za32, _s32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za32_s32_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za32_s32_1ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv4i32(i32 3, i32 [[TILESLICE]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za32_s32_1(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  uint32_t slice = slice_base + 3;
  SME_ACLE_FUNC(svwrite_ver_za32, _s32, _m)(3, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za64_s64(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za64_s64ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za64_s64(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za64, _s64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za64_s64_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za64_s64_1ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv2i64(i32 7, i32 [[TILESLICE]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za64_s64_1(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  uint32_t slice = slice_base + 1;
  SME_ACLE_FUNC(svwrite_ver_za64, _s64, _m)(7, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za8_u8(
// CHECK-CXX-LABEL: @_Z23test_svwrite_ver_za8_u8ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za8_u8(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za8, _u8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za8_u8_1(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za8_u8_1ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv16i8(i32 0, i32 [[TILESLICE]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za8_u8_1(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  uint32_t slice = slice_base + 15;
  SME_ACLE_FUNC(svwrite_ver_za8, _u8, _m)(0, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_u16(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za16_u16ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_u16(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za16, _u16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_u16_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za16_u16_1ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8i16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_u16_1(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_ver_za16, _u16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za32_u32(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za32_u32ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za32_u32(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za32, _u32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za32_u32_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za32_u32_1ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv4i32(i32 3, i32 [[TILESLICE]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za32_u32_1(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  uint32_t slice = slice_base + 3;
  SME_ACLE_FUNC(svwrite_ver_za32, _u32, _m)(3, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za64_u64(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za64_u64ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za64_u64(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za64, _u64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za64_u64_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za64_u64_1ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv2i64(i32 7, i32 [[TILESLICE]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za64_u64_1(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  uint32_t slice = slice_base + 1;
  SME_ACLE_FUNC(svwrite_ver_za64, _u64, _m)(7, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_f16(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za16_f16ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8f16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_f16(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za16, _f16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_f16_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za16_f16_1ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8f16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_f16_1(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_ver_za16, _f16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_bf16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za16_bf16ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8bf16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_bf16(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za16, _bf16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za16_bf16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za16_bf16_1ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv8bf16(i32 1, i32 [[TILESLICE]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za16_bf16_1(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  uint32_t slice = slice_base + 7;
  SME_ACLE_FUNC(svwrite_ver_za16, _bf16, _m)(1, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za32_f32(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za32_f32ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv4f32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za32_f32(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za32, _f32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za32_f32_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za32_f32_1ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv4f32(i32 3, i32 [[TILESLICE]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za32_f32_1(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  uint32_t slice = slice_base + 3;
  SME_ACLE_FUNC(svwrite_ver_za32, _f32, _m)(3, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za64_f64(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za64_f64ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv2f64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za64_f64(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za64, _f64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za64_f64_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za64_f64_1ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.write.vert.nxv2f64(i32 7, i32 [[TILESLICE]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za64_f64_1(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  uint32_t slice = slice_base + 1;
  SME_ACLE_FUNC(svwrite_ver_za64, _f64, _m)(7, slice, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s8(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za128_s8ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s8(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s8_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za128_s8_1ju10__SVBool_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv16i8(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s8_1(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s8, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_s16ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s16(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_s16_1ju10__SVBool_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8i16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s16_1(uint32_t slice_base, svbool_t pg, svint16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s32(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_s32ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s32(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s32_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_s32_1ju10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv4i32(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s32_1(uint32_t slice_base, svbool_t pg, svint32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s32, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s64(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_s64ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s64(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_s64_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_s64_1ju10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv2i64(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_s64_1(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _s64, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u8(
// CHECK-CXX-LABEL: @_Z25test_svwrite_ver_za128_u8ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv16i8(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u8(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u8, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u8_1(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za128_u8_1ju10__SVBool_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv16i8(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 16 x i1> [[PG:%.*]], <vscale x 16 x i8> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u8_1(uint32_t slice_base, svbool_t pg, svuint8_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u8, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_u16ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8i16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u16(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_u16_1ju10__SVBool_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8i16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i16> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u16_1(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u32(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_u32ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv4i32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u32(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u32_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_u32_1ju10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv4i32(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u32_1(uint32_t slice_base, svbool_t pg, svuint32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u32, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u64(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_u64ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv2i64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u64(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_u64_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_u64_1ju10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv2i64(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_u64_1(uint32_t slice_base, svbool_t pg, svuint64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _u64, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_f16(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_f16ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8f16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_f16(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _f16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_f16_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_f16_1ju10__SVBool_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8f16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x half> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_f16_1(uint32_t slice_base, svbool_t pg, svfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _f16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_bf16(
// CHECK-CXX-LABEL: @_Z27test_svwrite_ver_za128_bf16ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8bf16(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_bf16(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _bf16, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_bf16_1(
// CHECK-CXX-LABEL: @_Z29test_svwrite_ver_za128_bf16_1ju10__SVBool_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv8bf16(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 8 x i1> [[TMP0]], <vscale x 8 x bfloat> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_bf16_1(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _bf16, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_f32(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_f32ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv4f32(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_f32(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _f32, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_f32_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_f32_1ju10__SVBool_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv4f32(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 4 x i1> [[TMP0]], <vscale x 4 x float> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_f32_1(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _f32, _m)(15, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_f64(
// CHECK-CXX-LABEL: @_Z26test_svwrite_ver_za128_f64ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv2f64(i32 0, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_f64(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _f64, _m)(0, slice_base, pg, zn);
}

// CHECK-C-LABEL: @test_svwrite_ver_za128_f64_1(
// CHECK-CXX-LABEL: @_Z28test_svwrite_ver_za128_f64_1ju10__SVBool_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.writeq.vert.nxv2f64(i32 15, i32 [[SLICE_BASE:%.*]], <vscale x 2 x i1> [[TMP0]], <vscale x 2 x double> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svwrite_ver_za128_f64_1(uint32_t slice_base, svbool_t pg, svfloat64_t zn) {
  SME_ACLE_FUNC(svwrite_ver_za128, _f64, _m)(15, slice_base, pg, zn);
}
