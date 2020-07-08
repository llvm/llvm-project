// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svdupq_lane_s8(svint8_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %data, i64 %index)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_s8,,)(data, index);
}

svint16_t test_svdupq_lane_s16(svint16_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16> %data, i64 %index)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_s16,,)(data, index);
}

svint32_t test_svdupq_lane_s32(svint32_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %data, i64 %index)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_s32,,)(data, index);
}

svint64_t test_svdupq_lane_s64(svint64_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %data, i64 %index)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_s64,,)(data, index);
}

svuint8_t test_svdupq_lane_u8(svuint8_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %data, i64 %index)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_u8,,)(data, index);
}

svuint16_t test_svdupq_lane_u16(svuint16_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dupq.lane.nxv8i16(<vscale x 8 x i16> %data, i64 %index)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_u16,,)(data, index);
}

svuint32_t test_svdupq_lane_u32(svuint32_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dupq.lane.nxv4i32(<vscale x 4 x i32> %data, i64 %index)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_u32,,)(data, index);
}

svuint64_t test_svdupq_lane_u64(svuint64_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dupq.lane.nxv2i64(<vscale x 2 x i64> %data, i64 %index)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_u64,,)(data, index);
}

svfloat16_t test_svdupq_lane_f16(svfloat16_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.dupq.lane.nxv8f16(<vscale x 8 x half> %data, i64 %index)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_f16,,)(data, index);
}

svfloat32_t test_svdupq_lane_f32(svfloat32_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.dupq.lane.nxv4f32(<vscale x 4 x float> %data, i64 %index)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_f32,,)(data, index);
}

svfloat64_t test_svdupq_lane_f64(svfloat64_t data, uint64_t index)
{
  // CHECK-LABEL: test_svdupq_lane_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dupq.lane.nxv2f64(<vscale x 2 x double> %data, i64 %index)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svdupq_lane,_f64,,)(data, index);
}

svint8_t test_svdupq_n_s8(int8_t x0, int8_t x1, int8_t x2, int8_t x3,
                          int8_t x4, int8_t x5, int8_t x6, int8_t x7,
                          int8_t x8, int8_t x9, int8_t x10, int8_t x11,
                          int8_t x12, int8_t x13, int8_t x14, int8_t x15)
{
  // CHECK-LABEL: test_svdupq_n_s8
  // CHECK: %[[ALLOCA:.*]] = alloca [16 x i8], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i8 %x0, i8* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[ALLOCA]], i64 0, i64 15
  // CHECK: store i8 %x15, i8* %[[GEP]], align 1
  // CHECK-NOT: store
  // CHECK: %[[PTRUE:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %[[PTRUE]], i8* nonnull %[[BASE]])
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_s8,)(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
}

svint16_t test_svdupq_n_s16(int16_t x0, int16_t x1, int16_t x2, int16_t x3,
                            int16_t x4, int16_t x5, int16_t x6, int16_t x7)
{
  // CHECK-LABEL: test_svdupq_n_s16
  // CHECK-DAG: %[[ALLOCA:.*]] = alloca [8 x i16], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [8 x i16], [8 x i16]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i16 %x0, i16* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [8 x i16], [8 x i16]* %[[ALLOCA]], i64 0, i64 7
  // CHECK: store i16 %x7, i16* %[[GEP]], align 2
  // CHECK-NOT: store
  // CHECK: call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1rq.nxv8i16(<vscale x 8 x i1> %{{.*}}, i16* nonnull %[[BASE]])
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_s16,)(x0, x1, x2, x3, x4, x5, x6, x7);
}

svint32_t test_svdupq_n_s32(int32_t x0, int32_t x1, int32_t x2, int32_t x3)
{
  // CHECK-LABEL: test_svdupq_n_s32
  // CHECK: %[[ALLOCA:.*]] = alloca [4 x i32], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [4 x i32], [4 x i32]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i32 %x0, i32* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [4 x i32], [4 x i32]* %[[ALLOCA]], i64 0, i64 3
  // CHECK: store i32 %x3, i32* %[[GEP]], align 4
  // CHECK-NOT: store
  // CHECK: call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1rq.nxv4i32(<vscale x 4 x i1> %{{.*}}, i32* nonnull %[[BASE]])
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_s32,)(x0, x1, x2, x3);
}

svint64_t test_svdupq_n_s64(int64_t x0, int64_t x1)
{
  // CHECK-LABEL: test_svdupq_n_s64
  // CHECK: %[[ALLOCA:.*]] = alloca [2 x i64], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i64 %x0, i64* %[[BASE]], align 16
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 1
  // CHECK: store i64 %x1, i64* %[[GEP]], align 8
  // CHECK-NOT: store
  // CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1> %{{.*}}, i64* nonnull %[[BASE]])
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_s64,)(x0, x1);
}

svuint8_t test_svdupq_n_u8(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3,
                           uint8_t x4, uint8_t x5, uint8_t x6, uint8_t x7,
                           uint8_t x8, uint8_t x9, uint8_t x10, uint8_t x11,
                           uint8_t x12, uint8_t x13, uint8_t x14, uint8_t x15)
{
  // CHECK-LABEL: test_svdupq_n_u8
  // CHECK: %[[ALLOCA:.*]] = alloca [16 x i8], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i8 %x0, i8* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[ALLOCA]], i64 0, i64 15
  // CHECK: store i8 %x15, i8* %[[GEP]], align 1
  // CHECK-NOT: store
  // CHECK: %[[PTRUE:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %[[PTRUE]], i8* nonnull %[[BASE]])
  // CHECK: ret <vscale x 16 x i8> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_u8,)(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
}

svuint16_t test_svdupq_n_u16(uint16_t x0, uint16_t x1, uint16_t x2, uint16_t x3,
                             uint16_t x4, uint16_t x5, uint16_t x6, uint16_t x7)
{
  // CHECK-LABEL: test_svdupq_n_u16
  // CHECK: %[[ALLOCA:.*]] = alloca [8 x i16], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [8 x i16], [8 x i16]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i16 %x0, i16* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [8 x i16], [8 x i16]* %[[ALLOCA]], i64 0, i64 7
  // CHECK: store i16 %x7, i16* %[[GEP]], align 2
  // CHECK-NOT: store
  // CHECK: call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1rq.nxv8i16(<vscale x 8 x i1> %{{.*}}, i16* nonnull %[[BASE]])
  // CHECK: ret <vscale x 8 x i16> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_u16,)(x0, x1, x2, x3, x4, x5, x6, x7);
}

svuint32_t test_svdupq_n_u32(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3)
{
  // CHECK-LABEL: test_svdupq_n_u32
  // CHECK: %[[ALLOCA:.*]] = alloca [4 x i32], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [4 x i32], [4 x i32]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i32 %x0, i32* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [4 x i32], [4 x i32]* %[[ALLOCA]], i64 0, i64 3
  // CHECK: store i32 %x3, i32* %[[GEP]], align 4
  // CHECK-NOT: store
  // CHECK: call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1rq.nxv4i32(<vscale x 4 x i1> %{{.*}}, i32* nonnull %[[BASE]])
  // CHECK: ret <vscale x 4 x i32> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_u32,)(x0, x1, x2, x3);
}

svuint64_t test_svdupq_n_u64(uint64_t x0, uint64_t x1)
{
  // CHECK-LABEL: test_svdupq_n_u64
  // CHECK: %[[ALLOCA:.*]] = alloca [2 x i64], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i64 %x0, i64* %[[BASE]], align 16
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 1
  // CHECK: store i64 %x1, i64* %[[GEP]], align 8
  // CHECK-NOT: store
  // CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1> %{{.*}}, i64* nonnull %[[BASE]])
  // CHECK: ret <vscale x 2 x i64> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_u64,)(x0, x1);
}

svfloat16_t test_svdupq_n_f16(float16_t x0, float16_t x1, float16_t x2, float16_t x3,
                              float16_t x4, float16_t x5, float16_t x6, float16_t x7)
{
  // CHECK-LABEL: test_svdupq_n_f16
  // CHECK: %[[ALLOCA:.*]] = alloca [8 x half], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [8 x half], [8 x half]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store half %x0, half* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [8 x half], [8 x half]* %[[ALLOCA]], i64 0, i64 7
  // CHECK: store half %x7, half* %[[GEP]], align 2
  // CHECK-NOT: store
  // CHECK: call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ld1rq.nxv8f16(<vscale x 8 x i1> %{{.*}}, half* nonnull %[[BASE]])
  // CHECK: ret <vscale x 8 x half> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_f16,)(x0, x1, x2, x3, x4, x5, x6, x7);
}

svfloat32_t test_svdupq_n_f32(float32_t x0, float32_t x1, float32_t x2, float32_t x3)
{
  // CHECK-LABEL: test_svdupq_n_f32
  // CHECK: %[[ALLOCA:.*]] = alloca [4 x float], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [4 x float], [4 x float]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store float %x0, float* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [4 x float], [4 x float]* %[[ALLOCA]], i64 0, i64 3
  // CHECK: store float %x3, float* %[[GEP]], align 4
  // CHECK-NOT: store
  // CHECK: call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ld1rq.nxv4f32(<vscale x 4 x i1> %{{.*}}, float* nonnull %[[BASE]])
  // CHECK: ret <vscale x 4 x float> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_f32,)(x0, x1, x2, x3);
}

svfloat64_t test_svdupq_n_f64(float64_t x0, float64_t x1)
{
  // CHECK-LABEL: test_svdupq_n_f64
  // CHECK: %[[ALLOCA:.*]] = alloca [2 x double], align 16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [2 x double], [2 x double]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store double %x0, double* %[[BASE]], align 16
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [2 x double], [2 x double]* %[[ALLOCA]], i64 0, i64 1
  // CHECK: store double %x1, double* %[[GEP]], align 8
  // CHECK-NOT: store
  // CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ld1rq.nxv2f64(<vscale x 2 x i1> %{{.*}}, double* nonnull %[[BASE]])
  // CHECK: ret <vscale x 2 x double> %[[LOAD]]
  return SVE_ACLE_FUNC(svdupq,_n,_f64,)(x0, x1);
}

svbool_t test_svdupq_n_b8(bool x0, bool x1, bool x2, bool x3,
                          bool x4, bool x5, bool x6, bool x7,
                          bool x8, bool x9, bool x10, bool x11,
                          bool x12, bool x13, bool x14, bool x15)
{
  // CHECK-LABEL: test_svdupq_n_b8
  // CHECK-DAG: %[[ALLOCA:.*]] = alloca [16 x i8], align 16
  // CHECK-DAG: %[[X0:.*]] = zext i1 %x0 to i8
  // CHECK-DAG: %[[X15:.*]] = zext i1 %x15 to i8
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i8 %[[X0]], i8* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [16 x i8], [16 x i8]* %[[ALLOCA]], i64 0, i64 15
  // CHECK: store i8 %[[X15]], i8* %[[GEP]], align 1
  // CHECK-NOT: store
  // CHECK: %[[PTRUE:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1rq.nxv16i8(<vscale x 16 x i1> %[[PTRUE]], i8* nonnull %[[BASE]])
  // CHECK: %[[ZERO:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  // CHECK: %[[CMP:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %[[PTRUE]], <vscale x 16 x i8> %[[LOAD]], <vscale x 2 x i64> %[[ZERO]])
  // CHECK: ret <vscale x 16 x i1> %[[CMP]]
  return SVE_ACLE_FUNC(svdupq,_n,_b8,)(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
}

svbool_t test_svdupq_n_b16(bool x0, bool x1, bool x2, bool x3,
                           bool x4, bool x5, bool x6, bool x7)
{
  // CHECK-LABEL: test_svdupq_n_b16
  // CHECK-DAG: %[[ALLOCA:.*]] = alloca [8 x i16], align 16
  // CHECK-DAG: %[[X0:.*]] = zext i1 %x0 to i16
  // CHECK-DAG: %[[X7:.*]] = zext i1 %x7 to i16
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [8 x i16], [8 x i16]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i16 %[[X0]], i16* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [8 x i16], [8 x i16]* %[[ALLOCA]], i64 0, i64 7
  // CHECK: store i16 %[[X7]], i16* %[[GEP]], align 2
  // CHECK-NOT: store
  // CHECK: %[[PTRUE:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1rq.nxv8i16(<vscale x 8 x i1> %{{.*}}, i16* nonnull %[[BASE]])
  // CHECK: %[[ZERO:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  // CHECK: %[[CMP:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1> %{{.*}}, <vscale x 8 x i16> %[[LOAD]], <vscale x 2 x i64> %[[ZERO]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[CMP]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svdupq,_n,_b16,)(x0, x1, x2, x3, x4, x5, x6, x7);
}

svbool_t test_svdupq_n_b32(bool x0, bool x1, bool x2, bool x3)
{
  // CHECK-LABEL: test_svdupq_n_b32
  // CHECK-DAG: %[[ALLOCA:.*]] = alloca [4 x i32], align 16
  // CHECK-DAG: %[[X0:.*]] = zext i1 %x0 to i32
  // CHECK-DAG: %[[X3:.*]] = zext i1 %x3 to i32
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [4 x i32], [4 x i32]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i32 %[[X0]], i32* %[[BASE]], align 16
  // <assume other stores>
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [4 x i32], [4 x i32]* %[[ALLOCA]], i64 0, i64 3
  // CHECK: store i32 %[[X3]], i32* %[[GEP]], align 4
  // CHECK-NOT: store
  // CHECK: %[[PTRUE:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1rq.nxv4i32(<vscale x 4 x i1> %{{.*}}, i32* nonnull %[[BASE]])
  // CHECK: %[[ZERO:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %{{.*}}, <vscale x 4 x i32> %[[LOAD]], <vscale x 2 x i64> %[[ZERO]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svdupq,_n,_b32,)(x0, x1, x2, x3);
}

svbool_t test_svdupq_n_b64(bool x0, bool x1)
{
  // CHECK-LABEL: test_svdupq_n_b64
  // CHECK-DAG: %[[ALLOCA:.*]] = alloca [2 x i64], align 16
  // CHECK-DAG: %[[X0:.*]] = zext i1 %x0 to i64
  // CHECK-DAG: %[[X1:.*]] = zext i1 %x1 to i64
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i64 %[[X0]], i64* %[[BASE]], align 16
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 1
  // CHECK: store i64 %[[X1]], i64* %[[GEP]], align 8
  // CHECK-NOT: store
  // CHECK: %[[PTRUE:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1> %{{.*}}, i64* nonnull %[[BASE]])
  // CHECK: %[[ZERO:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 0)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %{{.*}}, <vscale x 2 x i64> %[[LOAD]], <vscale x 2 x i64> %[[ZERO]])
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return SVE_ACLE_FUNC(svdupq,_n,_b64,)(x0, x1);
}

// This test checks that the `alloca` is added to the entry-block.
svint64_t test_svdupq_control_flow(int64_t x0, int64_t x1, svint64_t Default, bool P)
{
  // CHECK-LABEL: test_svdupq_control_flow
  // CHECK: entry:
  // CHECK-DAG: %[[ALLOCA:.*]] = alloca [2 x i64], align 16
  // CHECK-DAG: [[BR:.*]]:
  // CHECK-DAG: %[[BASE:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 0
  // CHECK-DAG: store i64 %x0, i64* %[[BASE]], align 16
  // CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds [2 x i64], [2 x i64]* %[[ALLOCA]], i64 0, i64 1
  // CHECK: store i64 %x1, i64* %[[GEP]], align 8
  // CHECK-NOT: store
  // CHECK: call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1rq.nxv2i64(<vscale x 2 x i1> %{{.*}}, i64* nonnull %[[BASE]])
  // CHECK: [[END:.*]]:
  // CHECK: %[[RETVAL:.*]] = phi <vscale x 2 x i64> [ %[[LOAD]], %if.end ], [ %Default, %entry ]
  // CHECK: ret <vscale x 2 x i64> %[[RETVAL]]
  if (P)
    return Default;
  return SVE_ACLE_FUNC(svdupq,_n,_s64,)(x0, x1);
}
