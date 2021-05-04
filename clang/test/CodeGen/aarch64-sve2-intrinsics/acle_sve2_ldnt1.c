// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

svint32_t test_svldnt1_gather_u32base_s32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_s32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_s32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _s32, )(pg, bases);
}

svint64_t test_svldnt1_gather_u64base_s64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_s64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_s64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _s64, )(pg, bases);
}

svuint32_t test_svldnt1_gather_u32base_u32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_u32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_u32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _u32, )(pg, bases);
}

svuint64_t test_svldnt1_gather_u64base_u64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_u64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_u64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _u64, )(pg, bases);
}

svfloat32_t test_svldnt1_gather_u32base_f32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_f32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_f32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _f32, )(pg, bases);
}

svfloat64_t test_svldnt1_gather_u64base_f64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_f64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_f64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _f64, )(pg, bases);
}

svint64_t test_svldnt1_gather_s64offset_s64(svbool_t pg, const int64_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_s64offset_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_s64offset_s64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _s64)(pg, base, offsets);
}

svuint64_t test_svldnt1_gather_s64offset_u64(svbool_t pg, const uint64_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_s64offset_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_s64offset_u64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _u64)(pg, base, offsets);
}

svfloat64_t test_svldnt1_gather_s64offset_f64(svbool_t pg, const float64_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_s64offset_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_s64offset_f64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, s64, offset, _f64)(pg, base, offsets);
}

svint32_t test_svldnt1_gather_u32offset_s32(svbool_t pg, const int32_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_u32offset_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32offset_s32'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _s32)(pg, base, offsets);
}

svint64_t test_svldnt1_gather_u64offset_s64(svbool_t pg, const int64_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_u64offset_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64offset_s64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _s64)(pg, base, offsets);
}

svuint32_t test_svldnt1_gather_u32offset_u32(svbool_t pg, const uint32_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_u32offset_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i32(<vscale x 4 x i1> [[PG]], i32* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32offset_u32'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _u32)(pg, base, offsets);
}

svuint64_t test_svldnt1_gather_u64offset_u64(svbool_t pg, const uint64_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_u64offset_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64offset_u64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _u64)(pg, base, offsets);
}

svfloat32_t test_svldnt1_gather_u32offset_f32(svbool_t pg, const float32_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_u32offset_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4f32(<vscale x 4 x i1> [[PG]], float* %base, <vscale x 4 x i32> %offsets)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32offset_f32'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u32, offset, _f32)(pg, base, offsets);
}

svfloat64_t test_svldnt1_gather_u64offset_f64(svbool_t pg, const float64_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svldnt1_gather_u64offset_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %offsets)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64offset_f64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u64, offset, _f64)(pg, base, offsets);
}

svint32_t test_svldnt1_gather_u32base_offset_s32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_offset_s32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset_s32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_offset_s32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_s32, )(pg, bases, offset);
}

svint64_t test_svldnt1_gather_u64base_offset_s64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_offset_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset_s64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_offset_s64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_s64, )(pg, bases, offset);
}

svuint32_t test_svldnt1_gather_u32base_offset_u32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_offset_u32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset_u32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_offset_u32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_u32, )(pg, bases, offset);
}

svuint64_t test_svldnt1_gather_u64base_offset_u64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_offset_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset_u64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_offset_u64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_u64, )(pg, bases, offset);
}

svfloat32_t test_svldnt1_gather_u32base_offset_f32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_offset_f32
  // CHECK: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset_f32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_offset_f32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _offset_f32, )(pg, bases, offset);
}

svfloat64_t test_svldnt1_gather_u64base_offset_f64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_offset_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_offset_f64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_offset_f64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _offset_f64, )(pg, bases, offset);
}

svint64_t test_svldnt1_gather_s64index_s64(svbool_t pg, const int64_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svldnt1_gather_s64index_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_s64index_s64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _s64)(pg, base, indices);
}

svuint64_t test_svldnt1_gather_s64index_u64(svbool_t pg, const uint64_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svldnt1_gather_s64index_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_s64index_u64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _u64)(pg, base, indices);
}

svfloat64_t test_svldnt1_gather_s64index_f64(svbool_t pg, const float64_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svldnt1_gather_s64index_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.index.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_s64index_f64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, s64, index, _f64)(pg, base, indices);
}

svint64_t test_svldnt1_gather_u64index_s64(svbool_t pg, const int64_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svldnt1_gather_u64index_s64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64index_s64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _s64)(pg, base, indices);
}

svuint64_t test_svldnt1_gather_u64index_u64(svbool_t pg, const uint64_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svldnt1_gather_u64index_u64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.index.nxv2i64(<vscale x 2 x i1> [[PG]], i64* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64index_u64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _u64)(pg, base, indices);
}

svfloat64_t test_svldnt1_gather_u64index_f64(svbool_t pg, const float64_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svldnt1_gather_u64index_f64
  // CHECK: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.index.nxv2f64(<vscale x 2 x i1> [[PG]], double* %base, <vscale x 2 x i64> %indices)
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64index_f64'}}
  return SVE_ACLE_FUNC(svldnt1_gather_, u64, index, _f64)(pg, base, indices);
}

svint32_t test_svldnt1_gather_u32base_index_s32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_index_s32
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 2
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index_s32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_index_s32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_s32, )(pg, bases, index);
}

svint64_t test_svldnt1_gather_u64base_index_s64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_index_s64
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 3
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index_s64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_index_s64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_s64, )(pg, bases, index);
}

svuint32_t test_svldnt1_gather_u32base_index_u32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_index_u32
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 2
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 4 x i32> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index_u32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_index_u32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_u32, )(pg, bases, index);
}

svuint64_t test_svldnt1_gather_u64base_index_u64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_index_u64
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 3
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 2 x i64> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index_u64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_index_u64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_u64, )(pg, bases, index);
}

svfloat32_t test_svldnt1_gather_u32base_index_f32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svldnt1_gather_u32base_index_f32
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 2
  // CHECK: [[LOAD:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> [[PG]], <vscale x 4 x i32> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 4 x float> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index_f32'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u32base_index_f32'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u32base, _index_f32, )(pg, bases, index);
}

svfloat64_t test_svldnt1_gather_u64base_index_f64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svldnt1_gather_u64base_index_f64
  // CHECK-DAG: [[PG:%.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: [[SHL:%.*]] = shl i64 %index, 3
  // CHECK: [[LOAD:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> [[PG]], <vscale x 2 x i64> %bases, i64 [[SHL]])
  // CHECK: ret <vscale x 2 x double> [[LOAD]]
  // overload-warning@+2 {{implicit declaration of function 'svldnt1_gather_index_f64'}}
  // expected-warning@+1 {{implicit declaration of function 'svldnt1_gather_u64base_index_f64'}}
  return SVE_ACLE_FUNC(svldnt1_gather, _u64base, _index_f64, )(pg, bases, index);
}
