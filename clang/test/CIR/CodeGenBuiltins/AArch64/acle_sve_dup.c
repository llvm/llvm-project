// REQUIRES: aarch64-registered-target

// DEFINE: %{common_flags} = -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall
// DEFINE: %{optimize} = opt -passes=sroa -S

// RUN: %clang_cc1                        %{common_flags} -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=ALL,CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS %{common_flags} -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=ALL,CIR

// RUN: %clang_cc1                        %{common_flags} -fclangir -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS %{common_flags} -fclangir -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR

// RUN: %clang_cc1                        %{common_flags} -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS %{common_flags} -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR
#include <arm_sve.h>

#if defined __ARM_FEATURE_SME
#define MODE_ATTR __arm_streaming
#else
#define MODE_ATTR
#endif

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

//===------------------------------------------------------===//
// 1. UNPREDICTED SVDUP
//===------------------------------------------------------===//

// ALL-LABEL: @test_svdup_n_s8
svint8_t test_svdup_n_s8(int8_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!s8i) -> !cir.vector<[16] x !s8i>

// LLVM_OGCG_CIR-SAME: i8{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 16 x i8> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s8,)(op);
}

// ALL-LABEL: @test_svdup_n_s16
svint16_t test_svdup_n_s16(int16_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!s16i) -> !cir.vector<[8] x !s16i>

// LLVM_OGCG_CIR-SAME: i16{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 8 x i16> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s16,)(op);
}

// ALL-LABEL: @test_svdup_n_s32
svint32_t test_svdup_n_s32(int32_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!s32i) -> !cir.vector<[4] x !s32i>

// LLVM_OGCG_CIR-SAME: i32{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 4 x i32> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s32,)(op);
}

// ALL-LABEL: @test_svdup_n_s64
svint64_t test_svdup_n_s64(int64_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!s64i) -> !cir.vector<[2] x !s64i>

// LLVM_OGCG_CIR-SAME: i64{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 2 x i64> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s64,)(op);
}

// ALL-LABEL: @test_svdup_n_u8
svuint8_t test_svdup_n_u8(uint8_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u8i) -> !cir.vector<[16] x !u8i>

// LLVM_OGCG_CIR-SAME: i8{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 16 x i8> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u8,)(op);
}

// ALL-LABEL: @test_svdup_n_u16
svuint16_t test_svdup_n_u16(uint16_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u16i) -> !cir.vector<[8] x !u16i>

// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 8 x i16> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u16,)(op);
}

// ALL-LABEL: @test_svdup_n_u32
svuint32_t test_svdup_n_u32(uint32_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u32i) -> !cir.vector<[4] x !u32i>

// LLVM_OGCG_CIR-SAME: i32{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 4 x i32> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u32,)(op);
}

// ALL-LABEL: @test_svdup_n_u64
svuint64_t test_svdup_n_u64(uint64_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u64i) -> !cir.vector<[2] x !u64i>

// LLVM_OGCG_CIR-SAME: i64{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 2 x i64> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u64,)(op);
}

// ALL-LABEL: @test_svdup_n_f16
svfloat16_t test_svdup_n_f16(float16_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!cir.f16) -> !cir.vector<[8] x !cir.f16>

// LLVM_OGCG_CIR-SAME: half{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 8 x half> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f16,)(op);
}

// ALL-LABEL: @test_svdup_n_f32
svfloat32_t test_svdup_n_f32(float32_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!cir.float) -> !cir.vector<[4] x !cir.float>

// LLVM_OGCG_CIR-SAME: float{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 4 x float> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f32,)(op);
}

// ALL-LABEL: @test_svdup_n_f64
svfloat64_t test_svdup_n_f64(float64_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!cir.double) -> !cir.vector<[2] x !cir.double>

// LLVM_OGCG_CIR-SAME: double{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double [[OP]])
// LLVM_OGCG_CIR:    ret <vscale x 2 x double> [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f64,)(op);
}

//===------------------------------------------------------===//
// 2. PREDICATED ZERO-ING SVDUP
//===------------------------------------------------------===//

// ALL-LABEL: @test_svdup_n_s8_z
svint8_t test_svdup_n_s8_z(svbool_t pg, int8_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[16] x !s8i>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %{{.*}}, %{{.*}} :
// CIR-SAME:        -> !cir.vector<[16] x !s8i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i8{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> zeroinitializer, <vscale x 16 x i1> [[PG]], i8 [[OP]])
// LLVM_OGCG_CIR:     ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s8_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_s16_z(
svint16_t test_svdup_n_s16_z(svbool_t pg, int16_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[8] x !s16i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !s16i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i16{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> [[PG_CONVERTED]], i16 [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s16_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_s32_z(
svint32_t test_svdup_n_s32_z(svbool_t pg, int32_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[4] x !s32i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !s32i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i32{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> [[PG_CONVERTED]], i32 [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s32_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_s64_z(
svint64_t test_svdup_n_s64_z(svbool_t pg, int64_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[2] x !s64i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !s64i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i64{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> [[PG_CONVERTED]], i64 [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s64_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u8_z(
svuint8_t test_svdup_n_u8_z(svbool_t pg, uint8_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[16] x !u8i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %{{.*}}, %{{.*}} :
// CIR-SAME:        -> !cir.vector<[16] x !u8i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i8{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> zeroinitializer, <vscale x 16 x i1> [[PG]], i8 [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u8_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u16_z(
svuint16_t test_svdup_n_u16_z(svbool_t pg, uint16_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[8] x !u16i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !u16i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i16{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> [[PG_CONVERTED]], i16 [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u16_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u32_z(
svuint32_t test_svdup_n_u32_z(svbool_t pg, uint32_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[4] x !u32i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !u32i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i32{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> [[PG_CONVERTED]], i32 [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u32_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u64_z(
svuint64_t test_svdup_n_u64_z(svbool_t pg, uint64_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[2] x !u64i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !u64i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i64{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> [[PG_CONVERTED]], i64 [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u64_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_f16_z(
svfloat16_t test_svdup_n_f16_z(svbool_t pg, float16_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[8] x !cir.f16>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:        -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[8] x !cir.f16>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], half{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> zeroinitializer, <vscale x 8 x i1> [[PG_CONVERTED]], half [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f16_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_f32_z(
svfloat32_t test_svdup_n_f32_z(svbool_t pg, float32_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[4] x !cir.float>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.float>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], float{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> [[PG_CONVERTED]], float [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f32_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_f64_z(
svfloat64_t test_svdup_n_f64_z(svbool_t pg, float64_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[2] x !cir.double>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" %{{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.double>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], double{{.*}} [[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_CONVERTED:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG]])
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> [[PG_CONVERTED]], double [[OP]])
// LLVM_OGCG_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f64_z,)(pg, op);
}
