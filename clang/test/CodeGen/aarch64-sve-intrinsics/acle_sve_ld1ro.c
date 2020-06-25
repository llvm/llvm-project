// RUN: %clang_cc1 -D__ARM_FEATURE_SVE_MATMUL_FP64 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE_MATMUL_FP64 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svint8_t test_svld1ro_s8(svbool_t pg, const int8_t *base) {
  // CHECK-LABEL: test_svld1ro_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _s8, , )(pg, base);
}

svint16_t test_svld1ro_s16(svbool_t pg, const int16_t *base) {
  // CHECK-LABEL: test_svld1ro_s16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1ro.nxv8i16(<vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _s16, , )(pg, base);
}

svint32_t test_svld1ro_s32(svbool_t pg, const int32_t *base) {
  // CHECK-LABEL: test_svld1ro_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1ro.nxv4i32(<vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _s32, , )(pg, base);
}

svint64_t test_svld1ro_s64(svbool_t pg, const int64_t *base) {
  // CHECK-LABEL: test_svld1ro_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1ro.nxv2i64(<vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _s64, , )(pg, base);
}

svuint8_t test_svld1ro_u8(svbool_t pg, const uint8_t *base) {
  // CHECK-LABEL: test_svld1ro_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _u8, , )(pg, base);
}

svuint16_t test_svld1ro_u16(svbool_t pg, const uint16_t *base) {
  // CHECK-LABEL: test_svld1ro_u16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1ro.nxv8i16(<vscale x 8 x i1> %[[PG]], i16* %base)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _u16, , )(pg, base);
}

svuint32_t test_svld1ro_u32(svbool_t pg, const uint32_t *base) {
  // CHECK-LABEL: test_svld1ro_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1ro.nxv4i32(<vscale x 4 x i1> %[[PG]], i32* %base)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _u32, , )(pg, base);
}

svuint64_t test_svld1ro_u64(svbool_t pg, const uint64_t *base) {
  // CHECK-LABEL: test_svld1ro_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1ro.nxv2i64(<vscale x 2 x i1> %[[PG]], i64* %base)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _u64, , )(pg, base);
}

svfloat16_t test_svld1ro_f16(svbool_t pg, const float16_t *base) {
  // CHECK-LABEL: test_svld1ro_f16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.ld1ro.nxv8f16(<vscale x 8 x i1> %[[PG]], half* %base)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _f16, , )(pg, base);
}

svfloat32_t test_svld1ro_f32(svbool_t pg, const float32_t *base) {
  // CHECK-LABEL: test_svld1ro_f32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ld1ro.nxv4f32(<vscale x 4 x i1> %[[PG]], float* %base)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _f32, , )(pg, base);
}

svfloat64_t test_svld1ro_f64(svbool_t pg, const float64_t *base) {
  // CHECK-LABEL: test_svld1ro_f64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ld1ro.nxv2f64(<vscale x 2 x i1> %[[PG]], double* %base)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  return SVE_ACLE_FUNC(svld1ro, _f64, , )(pg, base);
}
