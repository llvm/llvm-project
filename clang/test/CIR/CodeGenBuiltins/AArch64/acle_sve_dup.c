// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=ALL,CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=ALL,CIR

// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR

// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64 -target-feature +sve -disable-O0-optnone -Werror -Wall -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,LLVM_OGCG_CIR
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

// ALL-LABEL: @test_svdup_n_s8
svint8_t test_svdup_n_s8(int8_t op) MODE_ATTR
{
// CIR-SAME:      %[[OP:.*]]: !s8i {{.*}} -> !cir.vector<[16] x !s8i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(1) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!s8i) -> !cir.vector<[16] x !s8i>

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
// CIR-SAME:      %[[OP:.*]]: !s16i {{.*}} -> !cir.vector<[8] x !s16i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(2) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!s16i) -> !cir.vector<[8] x !s16i>

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
// CIR-SAME:      %[[OP:.*]]: !s32i {{.*}} -> !cir.vector<[4] x !s32i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(4) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!s32i) -> !cir.vector<[4] x !s32i>

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
// CIR-SAME:      %[[OP:.*]]: !s64i {{.*}} -> !cir.vector<[2] x !s64i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(8) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!s64i) -> !cir.vector<[2] x !s64i>

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
// CIR-SAME:      %[[OP:.*]]: !u8i {{.*}} -> !cir.vector<[16] x !u8i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(1) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!u8i) -> !cir.vector<[16] x !u8i>

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
// CIR-SAME:      %[[OP:.*]]: !u16i {{.*}} -> !cir.vector<[8] x !u16i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(2) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!u16i) -> !cir.vector<[8] x !u16i>

// LLVM_OGCG_CIR-SAME: i16 {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca i16,{{([[:space:]]?i64 1,)?}} align 2
// LLVM_OGCG_CIR:    store i16 [[OP]], ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load i16, ptr [[OP_ADDR]], align 2
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_u16,)(op);
}

// ALL-LABEL: @test_svdup_n_u32
svuint32_t test_svdup_n_u32(uint32_t op) MODE_ATTR
{
// CIR-SAME:      %[[OP:.*]]: !u32i {{.*}} -> !cir.vector<[4] x !u32i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(4) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!u32i) -> !cir.vector<[4] x !u32i>

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
// CIR-SAME:      %[[OP:.*]]: !u64i {{.*}} -> !cir.vector<[2] x !u64i>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(8) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!u64i) -> !cir.vector<[2] x !u64i>

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
// CIR-SAME:      %[[OP:.*]]: !cir.f16 {{.*}} -> !cir.vector<[8] x !cir.f16>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(2) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!cir.f16) -> !cir.vector<[8] x !cir.f16>

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
// CIR-SAME:      %[[OP:.*]]: !cir.float {{.*}} -> !cir.vector<[4] x !cir.float>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(4) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!cir.float) -> !cir.vector<[4] x !cir.float>

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
// CIR-SAME:      %[[OP:.*]]: !cir.double {{.*}} -> !cir.vector<[2] x !cir.double>
// CIR:           %[[ALLOCA:.*]] = cir.alloca
// CIR:           cir.store %[[OP]], %[[ALLOCA]]
// CIR:           %[[LOAD:.*]] = cir.load align(8) %[[ALLOCA]]
// CIR:           cir.call_llvm_intrinsic "aarch64.sve.dup.x" %[[LOAD]] : (!cir.double) -> !cir.vector<[2] x !cir.double>

// LLVM_OGCG_CIR-SAME: double {{(noundef)?[[:space:]]?}}[[OP:%.*]])
// LLVM_OGCG_CIR:    [[OP_ADDR:%.*]] = alloca double,{{([[:space:]]?i64 1,)?}} align 8
// LLVM_OGCG_CIR:    store double [[OP]], ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[OP_LOAD:%.*]] = load double, ptr [[OP_ADDR]], align 8
// LLVM_OGCG_CIR:    [[RES:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double [[OP_LOAD]])
  return SVE_ACLE_FUNC(svdup,_n,_f64,)(op);
}
