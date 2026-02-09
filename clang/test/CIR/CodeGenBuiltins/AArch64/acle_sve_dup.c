// REQUIRES: aarch64-registered-target

// DEFINE: %{common_flags} = -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall
// DEFINE: %{optimize} = opt -O0 -S

// RUN: %clang_cc1                        %{common_flags} -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=ALL,CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS %{common_flags} -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=ALL,CIR

// RUN: %clang_cc1                        %{common_flags} -fclangir -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR,LLVM_VIA_CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS %{common_flags} -fclangir -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR,LLVM_VIA_CIR

// RUN: %clang_cc1                        %{common_flags} -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR,LLVM_DIRECT
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS %{common_flags} -emit-llvm -o - %s | %{optimize} | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR,LLVM_DIRECT
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

// LLVM_OGCG_CIR-SAME: i8 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i8,{{([[:space:]]?i64 1,)?}} align 1
// LLVM_OGCG_CIR:    store i8 [[OP]], ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i8, ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_s8,)(op);
}

// ALL-LABEL: @test_svdup_n_s16
svint16_t test_svdup_n_s16(int16_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!s16i) -> !cir.vector<[8] x !s16i>

// LLVM_OGCG_CIR-SAME: i16 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i16,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    store i16 [[OP]], ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i16, ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_s16,)(op);
}

// ALL-LABEL: @test_svdup_n_s32
svint32_t test_svdup_n_s32(int32_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!s32i) -> !cir.vector<[4] x !s32i>

// LLVM_OGCG_CIR-SAME: i32 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i32,{{([[:space:]]?i64 1,)?}} align 4
// LLVM_OGCG_CIR:    store i32 [[OP]], ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i32, ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_s32,)(op);
}

// ALL-LABEL: @test_svdup_n_s64
svint64_t test_svdup_n_s64(int64_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!s64i) -> !cir.vector<[2] x !s64i>

// LLVM_OGCG_CIR-SAME: i64 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i64,{{([[:space:]]?i64 1,)?}} align 8
// LLVM_OGCG_CIR:    store i64 [[OP]], ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i64, ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_s64,)(op);
}

// ALL-LABEL: @test_svdup_n_u8
svuint8_t test_svdup_n_u8(uint8_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u8i) -> !cir.vector<[16] x !u8i>

// LLVM_OGCG_CIR-SAME: i8 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i8,{{([[:space:]]?i64 1,)?}} align 1
// LLVM_OGCG_CIR:    store i8 [[OP]], ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i8, ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.x.nxv16i8(i8 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_u8,)(op);
}

// ALL-LABEL: @test_svdup_n_u16
svuint16_t test_svdup_n_u16(uint16_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u16i) -> !cir.vector<[8] x !u16i>

// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_u16,)(op);
}

// ALL-LABEL: @test_svdup_n_u32
svuint32_t test_svdup_n_u32(uint32_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u32i) -> !cir.vector<[4] x !u32i>

// LLVM_OGCG_CIR-SAME: i32 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i32,{{([[:space:]]?i64 1,)?}} align 4
// LLVM_OGCG_CIR:    store i32 [[OP]], ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i32, ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_u32,)(op);
}

// ALL-LABEL: @test_svdup_n_u64
svuint64_t test_svdup_n_u64(uint64_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!u64i) -> !cir.vector<[2] x !u64i>

// LLVM_OGCG_CIR-SAME: i64 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i64,{{([[:space:]]?i64 1,)?}} align 8
// LLVM_OGCG_CIR:    store i64 [[OP]], ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i64, ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_u64,)(op);
}

// ALL-LABEL: @test_svdup_n_f16
svfloat16_t test_svdup_n_f16(float16_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!cir.f16) -> !cir.vector<[8] x !cir.f16>

// LLVM_OGCG_CIR-SAME: half {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca half,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    store half [[OP]], ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load half, ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_f16,)(op);
}

// ALL-LABEL: @test_svdup_n_f32
svfloat32_t test_svdup_n_f32(float32_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!cir.float) -> !cir.vector<[4] x !cir.float>

// LLVM_OGCG_CIR-SAME: float {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca float,{{([[:space:]]?i64 1,)?}} align 4
// LLVM_OGCG_CIR:    store float [[OP]], ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load float, ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_f32,)(op);
}

// ALL-LABEL: @test_svdup_n_f64
svfloat64_t test_svdup_n_f64(float64_t op) MODE_ATTR
{
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %{{.*}} : (!cir.double) -> !cir.vector<[2] x !cir.double>

// LLVM_OGCG_CIR-SAME: double {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca double,{{([[:space:]]?i64 1,)?}} align 8
// LLVM_OGCG_CIR:    store double [[OP]], ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load double, ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_f64,)(op);
}

//===------------------------------------------------------===//
// 2. PREDICATED ZERO-ING SVDUP
//===------------------------------------------------------===//

// ALL-LABEL: @test_svdup_n_s8_z
svint8_t test_svdup_n_s8_z(svbool_t pg, int8_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[16] x !s8i>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], {{%.*}}, {{%.*}} :
// CIR-SAME:        -> !cir.vector<[16] x !s8i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i8 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i8,{{([[:space:]]?i64 1,)?}} align 1
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 16 x i8>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i8 [[OP]], ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i8, ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> zeroinitializer, <vscale x 16 x i1> [[TMP0]], i8 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP2]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP2]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s8_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_s16_z(
svint16_t test_svdup_n_s16_z(svbool_t pg, int16_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[8] x !s16i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{%.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !s16i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i16 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i16,{{([[:space:]]?i64 1,)?}} align 2
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 8 x i16>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i16 [[OP]], ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i16, ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> [[TMP2]], i16 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s16_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_s32_z(
svint32_t test_svdup_n_s32_z(svbool_t pg, int32_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[4] x !s32i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{%.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !s32i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i32 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i32,{{([[:space:]]?i64 1,)?}} align 4
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 4 x i32>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i32 [[OP]], ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i32, ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> [[TMP2]], i32 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s32_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_s64_z(
svint64_t test_svdup_n_s64_z(svbool_t pg, int64_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[2] x !s64i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !s64i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i64 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i64,{{([[:space:]]?i64 1,)?}} align 8
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 2 x i64>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i64 [[OP]], ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i64, ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> [[TMP2]], i64 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_s64_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u8_z(
svuint8_t test_svdup_n_u8_z(svbool_t pg, uint8_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[16] x !u8i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], {{.*}}, {{.*}} :
// CIR-SAME:        -> !cir.vector<[16] x !u8i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i8 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i8,{{([[:space:]]?i64 1,)?}} align 1
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 16 x i8>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i8 [[OP]], ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i8, ptr [[OP_ADDR]], align 1
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dup.nxv16i8(<vscale x 16 x i8> zeroinitializer, <vscale x 16 x i1> [[TMP0]], i8 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP2]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP2]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u8_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u16_z(
svuint16_t test_svdup_n_u16_z(svbool_t pg, uint16_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[8] x !u16i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:          -> !cir.vector<[8] x !u16i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i16 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i16,{{([[:space:]]?i64 1,)?}} align 2
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 8 x i16>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i16 [[OP]], ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i16, ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> zeroinitializer, <vscale x 8 x i1> [[TMP2]], i16 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u16_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u32_z(
svuint32_t test_svdup_n_u32_z(svbool_t pg, uint32_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[4] x !u32i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !u32i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i32 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i32,{{([[:space:]]?i64 1,)?}} align 4
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 4 x i32>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i32 [[OP]], ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i32, ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 4 x i1> [[TMP2]], i32 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u32_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_u64_z(
svuint64_t test_svdup_n_u64_z(svbool_t pg, uint64_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[2] x !u64i>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !u64i>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], i64 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i64,{{([[:space:]]?i64 1,)?}} align 8
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 2 x i64>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store i64 [[OP]], ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load i64, ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> zeroinitializer, <vscale x 2 x i1> [[TMP2]], i64 [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_u64_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_f16_z(
svfloat16_t test_svdup_n_f16_z(svbool_t pg, float16_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[8] x !cir.f16>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{.*}} :
// CIR-SAME:        -> !cir.vector<[8] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:        -> !cir.vector<[8] x !cir.f16>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], half {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca half,{{([[:space:]]?i64 1,)?}} align 2
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 8 x half>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store half [[OP]], ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load half, ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> zeroinitializer, <vscale x 8 x i1> [[TMP2]], half [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f16_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_f32_z(
svfloat32_t test_svdup_n_f32_z(svbool_t pg, float32_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[4] x !cir.float>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], {{.*}} :
// CIR-SAME:        -> !cir.vector<[4] x !cir.float>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], float {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca float,{{([[:space:]]?i64 1,)?}} align 4
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 4 x float>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store float [[OP]], ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load float, ptr [[OP_ADDR]], align 4
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dup.nxv4f32(<vscale x 4 x float> zeroinitializer, <vscale x 4 x i1> [[TMP2]], float [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f32_z,)(pg, op);
}

// ALL-LABEL: @test_svdup_n_f64_z(
svfloat64_t test_svdup_n_f64_z(svbool_t pg, float64_t op) MODE_ATTR
{
// CIR:           %[[CONST_0:.*]] = cir.const #cir.zero : !cir.vector<[2] x !cir.double>
// CIR:           %[[CONVERT_PG:.*]] = cir.call_llvm_intrinsic "aarch64.sve.convert.from.svbool" {{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.int<u, 1>>
// CIR:           %[[CALL_DUP:.*]] = cir.call_llvm_intrinsic "aarch64.sve.dup" %[[CONST_0]], %[[CONVERT_PG]], %{{.*}} :
// CIR-SAME:        -> !cir.vector<[2] x !cir.double>

// LLVM_OGCG_CIR-SAME: <vscale x 16 x i1> [[PG:%.*]], double {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[PG_ADDR:%.*]] = alloca <vscale x 16 x i1>,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca double,{{([[:space:]]?i64 1,)?}} align 8
//
// LLVM_VIA_CIR:    [[RES_ADDR:%.*]] = alloca <vscale x 2 x double>,{{([[:space:]]?i64 1,)?}} align 16
//
// LLVM_OGCG_CIR:    store <vscale x 16 x i1> [[PG]], ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    store double [[OP]], ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[PG_ADDR]], align 2
// LLVM_OGCG_CIR:    [[TMP1:%.*]] = load double, ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[TMP2:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[TMP0]])
// LLVM_OGCG_CIR:    [[TMP3:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> [[TMP2]], double [[TMP1]])
//
// LLVM_DIRECT:     ret {{.*}} [[TMP3]]
//
// LLVM_VIA_CIR:    store {{.*}} [[TMP3]], ptr [[RES_ADDR]]
// LLVM_VIA_CIR:    [[RES:%.*]] = load {{.*}} [[RES_ADDR]]
// LLVM_VIA_CIR:    ret {{.*}} [[RES]]
  return SVE_ACLE_FUNC(svdup,_n,_f64_z,)(pg, op);
}
