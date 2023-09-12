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

// CHECK-C-LABEL: @test_svread_hor_za8_s8(
// CHECK-CXX-LABEL: @_Z22test_svread_hor_za8_s8u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_hor_za8_s8(svint8_t zd, svbool_t pg, uint32_t slice_base) {
    return SME_ACLE_FUNC(svread_hor_za8, _s8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za8_s8_1(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za8_s8_1u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_hor_za8_s8_1(svint8_t zd, svbool_t pg, uint32_t slice_base) {
    uint32_t slice = slice_base + 15;
    return SME_ACLE_FUNC(svread_hor_za8, _s8, _m)(zd, pg, 0, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za16_s16(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za16_s16u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_hor_za16_s16(svint16_t zd, svbool_t pg, uint32_t slice_base) {
     return SME_ACLE_FUNC(svread_hor_za16, _s16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za16_s16_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za16_s16_1u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_hor_za16_s16_1(svint16_t zd, svbool_t pg, uint32_t slice_base) {
     uint32_t slice = slice_base + 7;
     return SME_ACLE_FUNC(svread_hor_za16, _s16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za32_s32(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za32_s32u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_hor_za32_s32(svint32_t zd, svbool_t pg, uint32_t slice_base) {
    return SME_ACLE_FUNC(svread_hor_za32, _s32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za32_s32_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za32_s32_1u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 3, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_hor_za32_s32_1(svint32_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 3;
    return SME_ACLE_FUNC(svread_hor_za32, _s32, _m)(zd, pg, 3, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za64_s64(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za64_s64u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_hor_za64_s64(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za64, _s64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za64_s64_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za64_s64_1u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 7, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_hor_za64_s64_1(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 1;
    return SME_ACLE_FUNC(svread_hor_za64, _s64, _m)(zd, pg, 7, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za8_u8(
// CHECK-CXX-LABEL: @_Z22test_svread_hor_za8_u8u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_hor_za8_u8(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za8, _u8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za8_u8_1(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za8_u8_1u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_hor_za8_u8_1(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 15;
    return SME_ACLE_FUNC(svread_hor_za8, _u8, _m)(zd, pg, 0, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za16_u16(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za16_u16u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_hor_za16_u16(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za16, _u16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za16_u16_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za16_u16_1u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_hor_za16_u16_1(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 7;
    return SME_ACLE_FUNC(svread_hor_za16, _u16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za32_u32(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za32_u32u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_hor_za32_u32(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za32, _u32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za32_u32_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za32_u32_1u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 3, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_hor_za32_u32_1(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 3;
    return SME_ACLE_FUNC(svread_hor_za32, _u32, _m)(zd, pg, 3, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za64_u64(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za64_u64u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_hor_za64_u64(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za64, _u64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za64_u64_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za64_u64_1u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 7, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_hor_za64_u64_1(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 1;
    return SME_ACLE_FUNC(svread_hor_za64, _u64, _m)(zd, pg, 7, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za16_f16(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za16_f16u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.read.horiz.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_hor_za16_f16(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za16, _f16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za16_f16_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za16_f16_1u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.read.horiz.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_hor_za16_f16_1(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 7;
    return SME_ACLE_FUNC(svread_hor_za16, _f16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za16_bf16(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za16_bf16u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.read.horiz.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_hor_za16_bf16(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za16, _bf16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za16_bf16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za16_bf16_1u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.read.horiz.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_hor_za16_bf16_1(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 7;
    return SME_ACLE_FUNC(svread_hor_za16, _bf16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za32_f32(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za32_f32u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.read.horiz.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_hor_za32_f32(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za32, _f32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za32_f32_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za32_f32_1u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.read.horiz.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 3, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_hor_za32_f32_1(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 3;
    return SME_ACLE_FUNC(svread_hor_za32, _f32, _m)(zd, pg, 3, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za64_f64(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za64_f64u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.read.horiz.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_hor_za64_f64(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za64, _f64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za64_f64_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za64_f64_1u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.read.horiz.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 7, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_hor_za64_f64_1(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 1;
    return SME_ACLE_FUNC(svread_hor_za64, _f64, _m)(zd, pg, 7, slice);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s8(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za128_s8u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_hor_za128_s8(svint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s8_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za128_s8_1u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_hor_za128_s8_1(svint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s8, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s16(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_s16u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_hor_za128_s16(svint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_s16_1u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_hor_za128_s16_1(svint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s32(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_s32u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_hor_za128_s32(svint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s32_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_s32_1u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_hor_za128_s32_1(svint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s32, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s64(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_s64u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_hor_za128_s64(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_s64_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_s64_1u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_hor_za128_s64_1(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _s64, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u8(
// CHECK-CXX-LABEL: @_Z24test_svread_hor_za128_u8u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_hor_za128_u8(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u8_1(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za128_u8_1u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.horiz.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_hor_za128_u8_1(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u8, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u16(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_u16u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_hor_za128_u16(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_u16_1u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.horiz.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_hor_za128_u16_1(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u32(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_u32u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_hor_za128_u32(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u32_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_u32_1u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.horiz.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_hor_za128_u32_1(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u32, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u64(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_u64u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_hor_za128_u64(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_u64_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_u64_1u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.horiz.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_hor_za128_u64_1(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _u64, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_f16(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_f16u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.readq.horiz.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_hor_za128_f16(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _f16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_f16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_f16_1u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.readq.horiz.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_hor_za128_f16_1(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _f16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_bf16(
// CHECK-CXX-LABEL: @_Z26test_svread_hor_za128_bf16u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.readq.horiz.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_hor_za128_bf16(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _bf16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_bf16_1(
// CHECK-CXX-LABEL: @_Z28test_svread_hor_za128_bf16_1u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.readq.horiz.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_hor_za128_bf16_1(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _bf16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_f32(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_f32u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.readq.horiz.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_hor_za128_f32(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _f32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_f32_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_f32_1u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.readq.horiz.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_hor_za128_f32_1(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _f32, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_f64(
// CHECK-CXX-LABEL: @_Z25test_svread_hor_za128_f64u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.readq.horiz.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_hor_za128_f64(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _f64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_hor_za128_f64_1(
// CHECK-CXX-LABEL: @_Z27test_svread_hor_za128_f64_1u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.readq.horiz.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_hor_za128_f64_1(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_hor_za128, _f64, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za8_s8(
// CHECK-CXX-LABEL: @_Z22test_svread_ver_za8_s8u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_ver_za8_s8(svint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za8, _s8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za8_s8_1(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za8_s8_1u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_ver_za8_s8_1(svint8_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 15;
    return SME_ACLE_FUNC(svread_ver_za8, _s8, _m)(zd, pg, 0, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za16_s16(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za16_s16u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_ver_za16_s16(svint16_t zd, svbool_t pg, uint32_t slice_base)  {
     return SME_ACLE_FUNC(svread_ver_za16, _s16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za16_s16_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za16_s16_1u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_ver_za16_s16_1(svint16_t zd, svbool_t pg, uint32_t slice_base)  {
     uint32_t slice = slice_base + 7;
     return SME_ACLE_FUNC(svread_ver_za16, _s16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za32_s32(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za32_s32u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_ver_za32_s32(svint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za32, _s32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za32_s32_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za32_s32_1u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 3, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_ver_za32_s32_1(svint32_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 3;
    return SME_ACLE_FUNC(svread_ver_za32, _s32, _m)(zd, pg, 3, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za64_s64(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za64_s64u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_ver_za64_s64(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za64, _s64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za64_s64_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za64_s64_1u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 7, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_ver_za64_s64_1(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 1;
    return SME_ACLE_FUNC(svread_ver_za64, _s64, _m)(zd, pg, 7, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za8_u8(
// CHECK-CXX-LABEL: @_Z22test_svread_ver_za8_u8u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_ver_za8_u8(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za8, _u8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za8_u8_1(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za8_u8_1u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.read.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_ver_za8_u8_1(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 15;
    return SME_ACLE_FUNC(svread_ver_za8, _u8, _m)(zd, pg, 0, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za16_u16(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za16_u16u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_ver_za16_u16(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za16, _u16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za16_u16_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za16_u16_1u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.read.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_ver_za16_u16_1(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 7;
    return SME_ACLE_FUNC(svread_ver_za16, _u16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za32_u32(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za32_u32u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_ver_za32_u32(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za32, _u32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za32_u32_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za32_u32_1u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.read.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 3, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_ver_za32_u32_1(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 3;
    return SME_ACLE_FUNC(svread_ver_za32, _u32, _m)(zd, pg, 3, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za64_u64(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za64_u64u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_ver_za64_u64(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za64, _u64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za64_u64_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za64_u64_1u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.read.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 7, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_ver_za64_u64_1(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 1;
    return SME_ACLE_FUNC(svread_ver_za64, _u64, _m)(zd, pg, 7, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za16_f16(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za16_f16u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.read.vert.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_ver_za16_f16(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za16, _f16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za16_f16_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za16_f16_1u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.read.vert.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_ver_za16_f16_1(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 7;
    return SME_ACLE_FUNC(svread_ver_za16, _f16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za16_bf16(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za16_bf16u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.read.vert.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_ver_za16_bf16(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za16, _bf16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za16_bf16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za16_bf16_1u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 7
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.read.vert.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 1, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_ver_za16_bf16_1(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 7;
    return SME_ACLE_FUNC(svread_ver_za16, _bf16, _m)(zd, pg, 1, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za32_f32(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za32_f32u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.read.vert.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_ver_za32_f32(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za32, _f32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za32_f32_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za32_f32_1u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 3
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.read.vert.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 3, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_ver_za32_f32_1(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 3;
    return SME_ACLE_FUNC(svread_ver_za32, _f32, _m)(zd, pg, 3, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za64_f64(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za64_f64u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.read.vert.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_ver_za64_f64(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za64, _f64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za64_f64_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za64_f64_1u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 1
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.read.vert.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 7, i32 [[TILESLICE]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_ver_za64_f64_1(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    uint32_t slice = slice_base + 1;
    return SME_ACLE_FUNC(svread_ver_za64, _f64, _m)(zd, pg, 7, slice);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s8(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za128_s8u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_ver_za128_s8(svint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s8_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za128_s8_1u10__SVInt8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svint8_t test_svread_ver_za128_s8_1(svint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s8, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s16(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_s16u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_ver_za128_s16(svint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_s16_1u11__SVInt16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svint16_t test_svread_ver_za128_s16_1(svint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s32(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_s32u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_ver_za128_s32(svint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s32_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_s32_1u11__SVInt32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svint32_t test_svread_ver_za128_s32_1(svint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s32, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s64(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_s64u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_ver_za128_s64(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_s64_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_s64_1u11__SVInt64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svint64_t test_svread_ver_za128_s64_1(svint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _s64, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u8(
// CHECK-CXX-LABEL: @_Z24test_svread_ver_za128_u8u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_ver_za128_u8(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u8, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u8_1(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za128_u8_1u11__SVUint8_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sme.readq.vert.nxv16i8(<vscale x 16 x i8> [[ZD:%.*]], <vscale x 16 x i1> [[PG:%.*]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 16 x i8> [[TMP0]]
//
svuint8_t test_svread_ver_za128_u8_1(svuint8_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u8, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u16(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_u16u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_ver_za128_u16(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_u16_1u12__SVUint16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i16> @llvm.aarch64.sme.readq.vert.nxv8i16(<vscale x 8 x i16> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
//
svuint16_t test_svread_ver_za128_u16_1(svuint16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u32(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_u32u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_ver_za128_u32(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u32_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_u32_1u12__SVUint32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i32> @llvm.aarch64.sme.readq.vert.nxv4i32(<vscale x 4 x i32> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP1]]
//
svuint32_t test_svread_ver_za128_u32_1(svuint32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u32, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u64(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_u64u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_ver_za128_u64(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_u64_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_u64_1u12__SVUint64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i64> @llvm.aarch64.sme.readq.vert.nxv2i64(<vscale x 2 x i64> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP1]]
//
svuint64_t test_svread_ver_za128_u64_1(svuint64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _u64, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_f16(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_f16u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.readq.vert.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_ver_za128_f16(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _f16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_f16_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_f16_1u13__SVFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x half> @llvm.aarch64.sme.readq.vert.nxv8f16(<vscale x 8 x half> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
//
svfloat16_t test_svread_ver_za128_f16_1(svfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _f16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_bf16(
// CHECK-CXX-LABEL: @_Z26test_svread_ver_za128_bf16u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.readq.vert.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_ver_za128_bf16(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _bf16, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_bf16_1(
// CHECK-CXX-LABEL: @_Z28test_svread_ver_za128_bf16_1u14__SVBFloat16_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x bfloat> @llvm.aarch64.sme.readq.vert.nxv8bf16(<vscale x 8 x bfloat> [[ZD:%.*]], <vscale x 8 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 8 x bfloat> [[TMP1]]
//
svbfloat16_t test_svread_ver_za128_bf16_1(svbfloat16_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _bf16, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_f32(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_f32u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.readq.vert.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_ver_za128_f32(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _f32, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_f32_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_f32_1u13__SVFloat32_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x float> @llvm.aarch64.sme.readq.vert.nxv4f32(<vscale x 4 x float> [[ZD:%.*]], <vscale x 4 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
//
svfloat32_t test_svread_ver_za128_f32_1(svfloat32_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _f32, _m)(zd, pg, 15, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_f64(
// CHECK-CXX-LABEL: @_Z25test_svread_ver_za128_f64u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.readq.vert.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_ver_za128_f64(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _f64, _m)(zd, pg, 0, slice_base);
}

// CHECK-C-LABEL: @test_svread_ver_za128_f64_1(
// CHECK-CXX-LABEL: @_Z27test_svread_ver_za128_f64_1u13__SVFloat64_tu10__SVBool_tj(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x double> @llvm.aarch64.sme.readq.vert.nxv2f64(<vscale x 2 x double> [[ZD:%.*]], <vscale x 2 x i1> [[TMP0]], i32 15, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
//
svfloat64_t test_svread_ver_za128_f64_1(svfloat64_t zd, svbool_t pg, uint32_t slice_base)  {
    return SME_ACLE_FUNC(svread_ver_za128, _f64, _m)(zd, pg, 15, slice_base);
}
