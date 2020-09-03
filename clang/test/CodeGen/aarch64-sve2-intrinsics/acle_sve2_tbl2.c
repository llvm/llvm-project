// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svtbl2_s8(svint8x2_t data, svuint8_t indices)
{
  // CHECK-LABEL: test_svtbl2_s8
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl2.nxv16i8(<vscale x 16 x i8> %[[V0]], <vscale x 16 x i8> %[[V1]], <vscale x 16 x i8> %indices)
  // CHECK-NEXT: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_s8'}}
  return SVE_ACLE_FUNC(svtbl2,_s8,,)(data, indices);
}

svint16_t test_svtbl2_s16(svint16x2_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbl2_s16
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl2.nxv8i16(<vscale x 8 x i16> %[[V0]], <vscale x 8 x i16> %[[V1]], <vscale x 8 x i16> %indices)
  // CHECK-NEXT: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_s16'}}
  return SVE_ACLE_FUNC(svtbl2,_s16,,)(data, indices);
}

svint32_t test_svtbl2_s32(svint32x2_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbl2_s32
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl2.nxv4i32(<vscale x 4 x i32> %[[V0]], <vscale x 4 x i32> %[[V1]], <vscale x 4 x i32> %indices)
  // CHECK-NEXT: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_s32'}}
  return SVE_ACLE_FUNC(svtbl2,_s32,,)(data, indices);
}

svint64_t test_svtbl2_s64(svint64x2_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbl2_s64
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl2.nxv2i64(<vscale x 2 x i64> %[[V0]], <vscale x 2 x i64> %[[V1]], <vscale x 2 x i64> %indices)
  // CHECK-NEXT: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_s64'}}
  return SVE_ACLE_FUNC(svtbl2,_s64,,)(data, indices);
}

svuint8_t test_svtbl2_u8(svuint8x2_t data, svuint8_t indices)
{
  // CHECK-LABEL: test_svtbl2_u8
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tuple.get.nxv16i8.nxv32i8(<vscale x 32 x i8> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbl2.nxv16i8(<vscale x 16 x i8> %[[V0]], <vscale x 16 x i8> %[[V1]], <vscale x 16 x i8> %indices)
  // CHECK-NEXT: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_u8'}}
  return SVE_ACLE_FUNC(svtbl2,_u8,,)(data, indices);
}

svuint16_t test_svtbl2_u16(svuint16x2_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbl2_u16
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tuple.get.nxv8i16.nxv16i16(<vscale x 16 x i16> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl2.nxv8i16(<vscale x 8 x i16> %[[V0]], <vscale x 8 x i16> %[[V1]], <vscale x 8 x i16> %indices)
  // CHECK-NEXT: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_u16'}}
  return SVE_ACLE_FUNC(svtbl2,_u16,,)(data, indices);
}

svuint32_t test_svtbl2_u32(svuint32x2_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbl2_u32
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tuple.get.nxv4i32.nxv8i32(<vscale x 8 x i32> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbl2.nxv4i32(<vscale x 4 x i32> %[[V0]], <vscale x 4 x i32> %[[V1]], <vscale x 4 x i32> %indices)
  // CHECK-NEXT: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_u32'}}
  return SVE_ACLE_FUNC(svtbl2,_u32,,)(data, indices);
}

svuint64_t test_svtbl2_u64(svuint64x2_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbl2_u64
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tuple.get.nxv2i64.nxv4i64(<vscale x 4 x i64> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbl2.nxv2i64(<vscale x 2 x i64> %[[V0]], <vscale x 2 x i64> %[[V1]], <vscale x 2 x i64> %indices)
  // CHECK-NEXT: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_u64'}}
  return SVE_ACLE_FUNC(svtbl2,_u64,,)(data, indices);
}

svfloat16_t test_svtbl2_f16(svfloat16x2_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbl2_f16
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv16f16(<vscale x 16 x half> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tuple.get.nxv8f16.nxv16f16(<vscale x 16 x half> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tbl2.nxv8f16(<vscale x 8 x half> %[[V0]], <vscale x 8 x half> %[[V1]], <vscale x 8 x i16> %indices)
  // CHECK-NEXT: ret <vscale x 8 x half> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_f16'}}
  return SVE_ACLE_FUNC(svtbl2,_f16,,)(data, indices);
}

svfloat32_t test_svtbl2_f32(svfloat32x2_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbl2_f32
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv8f32(<vscale x 8 x float> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tuple.get.nxv4f32.nxv8f32(<vscale x 8 x float> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tbl2.nxv4f32(<vscale x 4 x float> %[[V0]], <vscale x 4 x float> %[[V1]], <vscale x 4 x i32> %indices)
  // CHECK-NEXT: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_f32'}}
  return SVE_ACLE_FUNC(svtbl2,_f32,,)(data, indices);
}

svfloat64_t test_svtbl2_f64(svfloat64x2_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbl2_f64
  // CHECK-DAG: %[[V0:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv4f64(<vscale x 4 x double> %data, i32 0)
  // CHECK-DAG: %[[V1:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tuple.get.nxv2f64.nxv4f64(<vscale x 4 x double> %data, i32 1)
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tbl2.nxv2f64(<vscale x 2 x double> %[[V0]], <vscale x 2 x double> %[[V1]], <vscale x 2 x i64> %indices)
  // CHECK-NEXT: ret <vscale x 2 x double> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbl2'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbl2_f64'}}
  return SVE_ACLE_FUNC(svtbl2,_f64,,)(data, indices);
}
