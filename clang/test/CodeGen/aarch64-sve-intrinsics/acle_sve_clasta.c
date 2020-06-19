// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
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

svint8_t test_svclasta_s8(svbool_t pg, svint8_t fallback, svint8_t data)
{
  // CHECK-LABEL: test_svclasta_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.clasta.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %fallback, <vscale x 16 x i8> %data)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_s8,,)(pg, fallback, data);
}

svint16_t test_svclasta_s16(svbool_t pg, svint16_t fallback, svint16_t data)
{
  // CHECK-LABEL: test_svclasta_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.clasta.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %fallback, <vscale x 8 x i16> %data)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_s16,,)(pg, fallback, data);
}

svint32_t test_svclasta_s32(svbool_t pg, svint32_t fallback, svint32_t data)
{
  // CHECK-LABEL: test_svclasta_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.clasta.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %fallback, <vscale x 4 x i32> %data)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_s32,,)(pg, fallback, data);
}

svint64_t test_svclasta_s64(svbool_t pg, svint64_t fallback, svint64_t data)
{
  // CHECK-LABEL: test_svclasta_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.clasta.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %fallback, <vscale x 2 x i64> %data)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_s64,,)(pg, fallback, data);
}

svuint8_t test_svclasta_u8(svbool_t pg, svuint8_t fallback, svuint8_t data)
{
  // CHECK-LABEL: test_svclasta_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.clasta.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %fallback, <vscale x 16 x i8> %data)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_u8,,)(pg, fallback, data);
}

svuint16_t test_svclasta_u16(svbool_t pg, svuint16_t fallback, svuint16_t data)
{
  // CHECK-LABEL: test_svclasta_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.clasta.nxv8i16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x i16> %fallback, <vscale x 8 x i16> %data)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_u16,,)(pg, fallback, data);
}

svuint32_t test_svclasta_u32(svbool_t pg, svuint32_t fallback, svuint32_t data)
{
  // CHECK-LABEL: test_svclasta_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.clasta.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %fallback, <vscale x 4 x i32> %data)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_u32,,)(pg, fallback, data);
}

svuint64_t test_svclasta_u64(svbool_t pg, svuint64_t fallback, svuint64_t data)
{
  // CHECK-LABEL: test_svclasta_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.clasta.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %fallback, <vscale x 2 x i64> %data)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_u64,,)(pg, fallback, data);
}

svfloat16_t test_svclasta_f16(svbool_t pg, svfloat16_t fallback, svfloat16_t data)
{
  // CHECK-LABEL: test_svclasta_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.clasta.nxv8f16(<vscale x 8 x i1> %[[PG]], <vscale x 8 x half> %fallback, <vscale x 8 x half> %data)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_f16,,)(pg, fallback, data);
}

svfloat32_t test_svclasta_f32(svbool_t pg, svfloat32_t fallback, svfloat32_t data)
{
  // CHECK-LABEL: test_svclasta_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.clasta.nxv4f32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x float> %fallback, <vscale x 4 x float> %data)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_f32,,)(pg, fallback, data);
}

svfloat64_t test_svclasta_f64(svbool_t pg, svfloat64_t fallback, svfloat64_t data)
{
  // CHECK-LABEL: test_svclasta_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.clasta.nxv2f64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x double> %fallback, <vscale x 2 x double> %data)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_f64,,)(pg, fallback, data);
}

int8_t test_svclasta_n_s8(svbool_t pg, int8_t fallback, svint8_t data)
{
  // CHECK-LABEL: test_svclasta_n_s8
  // CHECK: %[[INTRINSIC:.*]] = call i8 @llvm.aarch64.sve.clasta.n.nxv16i8(<vscale x 16 x i1> %pg, i8 %fallback, <vscale x 16 x i8> %data)
  // CHECK: ret i8 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_s8,,)(pg, fallback, data);
}

int16_t test_svclasta_n_s16(svbool_t pg, int16_t fallback, svint16_t data)
{
  // CHECK-LABEL: test_svclasta_n_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i16 @llvm.aarch64.sve.clasta.n.nxv8i16(<vscale x 8 x i1> %[[PG]], i16 %fallback, <vscale x 8 x i16> %data)
  // CHECK: ret i16 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_s16,,)(pg, fallback, data);
}

int32_t test_svclasta_n_s32(svbool_t pg, int32_t fallback, svint32_t data)
{
  // CHECK-LABEL: test_svclasta_n_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.clasta.n.nxv4i32(<vscale x 4 x i1> %[[PG]], i32 %fallback, <vscale x 4 x i32> %data)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_s32,,)(pg, fallback, data);
}

int64_t test_svclasta_n_s64(svbool_t pg, int64_t fallback, svint64_t data)
{
  // CHECK-LABEL: test_svclasta_n_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.clasta.n.nxv2i64(<vscale x 2 x i1> %[[PG]], i64 %fallback, <vscale x 2 x i64> %data)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_s64,,)(pg, fallback, data);
}

uint8_t test_svclasta_n_u8(svbool_t pg, uint8_t fallback, svuint8_t data)
{
  // CHECK-LABEL: test_svclasta_n_u8
  // CHECK: %[[INTRINSIC:.*]] = call i8 @llvm.aarch64.sve.clasta.n.nxv16i8(<vscale x 16 x i1> %pg, i8 %fallback, <vscale x 16 x i8> %data)
  // CHECK: ret i8 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_u8,,)(pg, fallback, data);
}

uint16_t test_svclasta_n_u16(svbool_t pg, uint16_t fallback, svuint16_t data)
{
  // CHECK-LABEL: test_svclasta_n_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i16 @llvm.aarch64.sve.clasta.n.nxv8i16(<vscale x 8 x i1> %[[PG]], i16 %fallback, <vscale x 8 x i16> %data)
  // CHECK: ret i16 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_u16,,)(pg, fallback, data);
}

uint32_t test_svclasta_n_u32(svbool_t pg, uint32_t fallback, svuint32_t data)
{
  // CHECK-LABEL: test_svclasta_n_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i32 @llvm.aarch64.sve.clasta.n.nxv4i32(<vscale x 4 x i1> %[[PG]], i32 %fallback, <vscale x 4 x i32> %data)
  // CHECK: ret i32 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_u32,,)(pg, fallback, data);
}

uint64_t test_svclasta_n_u64(svbool_t pg, uint64_t fallback, svuint64_t data)
{
  // CHECK-LABEL: test_svclasta_n_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.clasta.n.nxv2i64(<vscale x 2 x i1> %[[PG]], i64 %fallback, <vscale x 2 x i64> %data)
  // CHECK: ret i64 %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_u64,,)(pg, fallback, data);
}

float16_t test_svclasta_n_f16(svbool_t pg, float16_t fallback, svfloat16_t data)
{
  // CHECK-LABEL: test_svclasta_n_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call half @llvm.aarch64.sve.clasta.n.nxv8f16(<vscale x 8 x i1> %[[PG]], half %fallback, <vscale x 8 x half> %data)
  // CHECK: ret half %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_f16,,)(pg, fallback, data);
}

float32_t test_svclasta_n_f32(svbool_t pg, float32_t fallback, svfloat32_t data)
{
  // CHECK-LABEL: test_svclasta_n_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call float @llvm.aarch64.sve.clasta.n.nxv4f32(<vscale x 4 x i1> %[[PG]], float %fallback, <vscale x 4 x float> %data)
  // CHECK: ret float %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_f32,,)(pg, fallback, data);
}

float64_t test_svclasta_n_f64(svbool_t pg, float64_t fallback, svfloat64_t data)
{
  // CHECK-LABEL: test_svclasta_n_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call double @llvm.aarch64.sve.clasta.n.nxv2f64(<vscale x 2 x i1> %[[PG]], double %fallback, <vscale x 2 x double> %data)
  // CHECK: ret double %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svclasta,_n_f64,,)(pg, fallback, data);
}
