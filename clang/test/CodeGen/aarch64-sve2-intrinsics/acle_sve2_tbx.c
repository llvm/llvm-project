// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svtbx_s8(svint8_t fallback, svint8_t data, svuint8_t indices)
{
  // CHECK-LABEL: test_svtbx_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbx.nxv16i8(<vscale x 16 x i8> %fallback, <vscale x 16 x i8> %data, <vscale x 16 x i8> %indices)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_s8'}}
  return SVE_ACLE_FUNC(svtbx,_s8,,)(fallback, data, indices);
}

svint16_t test_svtbx_s16(svint16_t fallback, svint16_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbx_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbx.nxv8i16(<vscale x 8 x i16> %fallback, <vscale x 8 x i16> %data, <vscale x 8 x i16> %indices)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_s16'}}
  return SVE_ACLE_FUNC(svtbx,_s16,,)(fallback, data, indices);
}

svint32_t test_svtbx_s32(svint32_t fallback, svint32_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbx_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbx.nxv4i32(<vscale x 4 x i32> %fallback, <vscale x 4 x i32> %data, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_s32'}}
  return SVE_ACLE_FUNC(svtbx,_s32,,)(fallback, data, indices);
}

svint64_t test_svtbx_s64(svint64_t fallback, svint64_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbx_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbx.nxv2i64(<vscale x 2 x i64> %fallback, <vscale x 2 x i64> %data, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_s64'}}
  return SVE_ACLE_FUNC(svtbx,_s64,,)(fallback, data, indices);
}

svuint8_t test_svtbx_u8(svuint8_t fallback, svuint8_t data, svuint8_t indices)
{
  // CHECK-LABEL: test_svtbx_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbx.nxv16i8(<vscale x 16 x i8> %fallback, <vscale x 16 x i8> %data, <vscale x 16 x i8> %indices)
  // CHECK: ret <vscale x 16 x i8> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_u8'}}
  return SVE_ACLE_FUNC(svtbx,_u8,,)(fallback, data, indices);
}

svuint16_t test_svtbx_u16(svuint16_t fallback, svuint16_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbx_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.tbx.nxv8i16(<vscale x 8 x i16> %fallback, <vscale x 8 x i16> %data, <vscale x 8 x i16> %indices)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_u16'}}
  return SVE_ACLE_FUNC(svtbx,_u16,,)(fallback, data, indices);
}

svuint32_t test_svtbx_u32(svuint32_t fallback, svuint32_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbx_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.tbx.nxv4i32(<vscale x 4 x i32> %fallback, <vscale x 4 x i32> %data, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_u32'}}
  return SVE_ACLE_FUNC(svtbx,_u32,,)(fallback, data, indices);
}

svuint64_t test_svtbx_u64(svuint64_t fallback, svuint64_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbx_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.tbx.nxv2i64(<vscale x 2 x i64> %fallback, <vscale x 2 x i64> %data, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_u64'}}
  return SVE_ACLE_FUNC(svtbx,_u64,,)(fallback, data, indices);
}

svfloat16_t test_svtbx_f16(svfloat16_t fallback, svfloat16_t data, svuint16_t indices)
{
  // CHECK-LABEL: test_svtbx_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.tbx.nxv8f16(<vscale x 8 x half> %fallback, <vscale x 8 x half> %data, <vscale x 8 x i16> %indices)
  // CHECK: ret <vscale x 8 x half> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_f16'}}
  return SVE_ACLE_FUNC(svtbx,_f16,,)(fallback, data, indices);
}

svfloat32_t test_svtbx_f32(svfloat32_t fallback, svfloat32_t data, svuint32_t indices)
{
  // CHECK-LABEL: test_svtbx_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.tbx.nxv4f32(<vscale x 4 x float> %fallback, <vscale x 4 x float> %data, <vscale x 4 x i32> %indices)
  // CHECK: ret <vscale x 4 x float> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_f32'}}
  return SVE_ACLE_FUNC(svtbx,_f32,,)(fallback, data, indices);
}

svfloat64_t test_svtbx_f64(svfloat64_t fallback, svfloat64_t data, svuint64_t indices)
{
  // CHECK-LABEL: test_svtbx_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.tbx.nxv2f64(<vscale x 2 x double> %fallback, <vscale x 2 x double> %data, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x double> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svtbx'}}
  // expected-warning@+1 {{implicit declaration of function 'svtbx_f64'}}
  return SVE_ACLE_FUNC(svtbx,_f64,,)(fallback, data, indices);
}
