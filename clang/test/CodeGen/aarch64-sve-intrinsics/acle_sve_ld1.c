// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s -D__ARM_FEATURE_SVE | FileCheck %s

#include <arm_sve.h>
//
// ld1
//

svint8_t test_svld1_s8(svbool_t pg, const int8_t *base)
{
  // CHECK-LABEL: test_svld1_s8
  // CHECK: <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* %{{.*}}, i32 1, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i8> zeroinitializer)
  return svld1_s8(pg, base);
}

svint16_t test_svld1_s16(svbool_t pg, const int16_t *base)
{
  // CHECK-LABEL: test_svld1_s16
  // CHECK: <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* %{{.*}}, i32 1, <vscale x 8 x i1> %{{.*}}, <vscale x 8 x i16> zeroinitializer)
  return svld1_s16(pg, base);
}

svint32_t test_svld1_s32(svbool_t pg, const int32_t *base)
{
  // CHECK-LABEL: test_svld1_s32
  // CHECK: <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %{{.*}}, i32 1, <vscale x 4 x i1> %{{.*}}, <vscale x 4 x i32> zeroinitializer)
  return svld1_s32(pg, base);
}

svint64_t test_svld1_s64(svbool_t pg, const int64_t *base)
{
  // CHECK-LABEL: test_svld1_s64
  // CHECK: <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* %{{.*}}, i32 1, <vscale x 2 x i1> %{{.*}}, <vscale x 2 x i64> zeroinitializer)
  return svld1_s64(pg, base);
}

svuint8_t test_svld1_u8(svbool_t pg, const uint8_t *base)
{
  // CHECK-LABEL: test_svld1_u8
  // CHECK: <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* %{{.*}}, i32 1, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i8> zeroinitializer)
  return svld1_u8(pg, base);
}

svuint16_t test_svld1_u16(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svld1_u16
  // CHECK: <vscale x 8 x i16> @llvm.masked.load.nxv8i16.p0nxv8i16(<vscale x 8 x i16>* %{{.*}}, i32 1, <vscale x 8 x i1> %{{.*}}, <vscale x 8 x i16> zeroinitializer)
  return svld1_u16(pg, base);
}

svuint32_t test_svld1_u32(svbool_t pg, const uint32_t *base)
{
  // CHECK-LABEL: test_svld1_u32
  // CHECK: <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %{{.*}}, i32 1, <vscale x 4 x i1> %{{.*}}, <vscale x 4 x i32> zeroinitializer)
  return svld1_u32(pg, base);
}

svuint64_t test_svld1_u64(svbool_t pg, const uint64_t *base)
{
  // CHECK-LABEL: test_svld1_u64
  // CHECK: <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* %{{.*}}, i32 1, <vscale x 2 x i1> %{{.*}}, <vscale x 2 x i64> zeroinitializer)
  return svld1_u64(pg, base);
}

svfloat16_t test_svld1_f16(svbool_t pg, const float16_t *base)
{
  // CHECK-LABEL: test_svld1_f16
  // CHECK: <vscale x 8 x half> @llvm.masked.load.nxv8f16.p0nxv8f16(<vscale x 8 x half>* %{{.*}}, i32 1, <vscale x 8 x i1> %{{.*}}, <vscale x 8 x half> zeroinitializer)
  return svld1_f16(pg, base);
}

svfloat32_t test_svld1_f32(svbool_t pg, const float32_t *base)
{
  // CHECK-LABEL: test_svld1_f32
  // CHECK: <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* %{{.*}}, i32 1, <vscale x 4 x i1> %{{.*}}, <vscale x 4 x float> zeroinitializer)
  return svld1_f32(pg, base);
}

svfloat64_t test_svld1_f64(svbool_t pg, const float64_t *base)
{
  // CHECK-LABEL: test_svld1_f64
  // CHECK: <vscale x 2 x double> @llvm.masked.load.nxv2f64.p0nxv2f64(<vscale x 2 x double>* %{{.*}}, i32 1, <vscale x 2 x i1> %{{.*}}, <vscale x 2 x double> zeroinitializer)
  return svld1_f64(pg, base);
}
