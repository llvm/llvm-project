// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s
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

svint32_t test_svld1uh_s32(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svld1uh_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return svld1uh_s32(pg, base);
}

svint64_t test_svld1uh_s64(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svld1uh_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return svld1uh_s64(pg, base);
}

svuint32_t test_svld1uh_u32(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svld1uh_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return svld1uh_u32(pg, base);
}

svuint64_t test_svld1uh_u64(svbool_t pg, const uint16_t *base)
{
  // CHECK-LABEL: test_svld1uh_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return svld1uh_u64(pg, base);
}

svint32_t test_svld1uh_vnum_s32(svbool_t pg, const uint16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1uh_vnum_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return svld1uh_vnum_s32(pg, base, vnum);
}

svint64_t test_svld1uh_vnum_s64(svbool_t pg, const uint16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1uh_vnum_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return svld1uh_vnum_s64(pg, base, vnum);
}

svuint32_t test_svld1uh_vnum_u32(svbool_t pg, const uint16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1uh_vnum_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 4 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return svld1uh_vnum_u32(pg, base, vnum);
}

svuint64_t test_svld1uh_vnum_u64(svbool_t pg, const uint16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld1uh_vnum_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast i16* %base to <vscale x 2 x i16>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %[[GEP]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return svld1uh_vnum_u64(pg, base, vnum);
}

svint32_t test_svld1uh_gather_u32base_s32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svld1uh_gather_u32base_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u32base, _s32, )(pg, bases);
}

svint64_t test_svld1uh_gather_u64base_s64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svld1uh_gather_u64base_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u64base, _s64, )(pg, bases);
}

svuint32_t test_svld1uh_gather_u32base_u32(svbool_t pg, svuint32_t bases) {
  // CHECK-LABEL: test_svld1uh_gather_u32base_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 0)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: et <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u32base, _u32, )(pg, bases);
}

svuint64_t test_svld1uh_gather_u64base_u64(svbool_t pg, svuint64_t bases) {
  // CHECK-LABEL: test_svld1uh_gather_u64base_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 0)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u64base, _u64, )(pg, bases);
}

svint32_t test_svld1uh_gather_s32offset_s32(svbool_t pg, const uint16_t *base, svint32_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_s32offset_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s32, offset_s32, )(pg, base, offsets);
}

svint64_t test_svld1uh_gather_s64offset_s64(svbool_t pg, const uint16_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_s64offset_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %1 to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s64, offset_s64, )(pg, base, offsets);
}

svuint32_t test_svld1uh_gather_s32offset_u32(svbool_t pg, const uint16_t *base, svint32_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_s32offset_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s32, offset_u32, )(pg, base, offsets);
}

svuint64_t test_svld1uh_gather_s64offset_u64(svbool_t pg, const uint16_t *base, svint64_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_s64offset_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %1 to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s64, offset_u64, )(pg, base, offsets);
}

svint32_t test_svld1uh_gather_u32offset_s32(svbool_t pg, const uint16_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_u32offset_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u32, offset_s32, )(pg, base, offsets);
}

svint64_t test_svld1uh_gather_u64offset_s64(svbool_t pg, const uint16_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_u64offset_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u64, offset_s64, )(pg, base, offsets);
}

svuint32_t test_svld1uh_gather_u32offset_u32(svbool_t pg, const uint16_t *base, svuint32_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_u32offset_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u32, offset_u32, )(pg, base, offsets);
}

svuint64_t test_svld1uh_gather_u64offset_u64(svbool_t pg, const uint16_t *base, svuint64_t offsets) {
  // CHECK-LABEL: test_svld1uh_gather_u64offset_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %offsets)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %1 to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u64, offset_u64, )(pg, base, offsets);
}

svint32_t test_svld1uh_gather_u32base_offset_s32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svld1uh_gather_u32base_offset_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u32base, _offset_s32, )(pg, bases, offset);
}

svint64_t test_svld1uh_gather_u64base_offset_s64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svld1uh_gather_u64base_offset_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u64base, _offset_s64, )(pg, bases, offset);
}

svuint32_t test_svld1uh_gather_u32base_offset_u32(svbool_t pg, svuint32_t bases, int64_t offset) {
  // CHECK-LABEL: test_svld1uh_gather_u32base_offset_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %offset)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %2
  return SVE_ACLE_FUNC(svld1uh_gather, _u32base, _offset_u32, )(pg, bases, offset);
}

svuint64_t test_svld1uh_gather_u64base_offset_u64(svbool_t pg, svuint64_t bases, int64_t offset) {
  // CHECK-LABEL: test_svld1uh_gather_u64base_offset_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %offset)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u64base, _offset_u64, )(pg, bases, offset);
}

svint32_t test_svld1uh_gather_s32index_s32(svbool_t pg, const uint16_t *base, svint32_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_s32index_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s32, index_s32, )(pg, base, indices);
}

svint64_t test_svld1uh_gather_s64index_s64(svbool_t pg, const uint16_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_s64index_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s64, index_s64, )(pg, base, indices);
}

svuint32_t test_svld1uh_gather_s32index_u32(svbool_t pg, const uint16_t *base, svint32_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_s32index_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.sxtw.index.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s32, index_u32, )(pg, base, indices);
}

svuint64_t test_svld1uh_gather_s64index_u64(svbool_t pg, const uint16_t *base, svint64_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_s64index_u64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, s64, index_u64, )(pg, base, indices);
}

svint32_t test_svld1uh_gather_u32index_s32(svbool_t pg, const uint16_t *base, svuint32_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_u32index_s32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u32, index_s32, )(pg, base, indices);
}

svint64_t test_svld1uh_gather_u64index_s64(svbool_t pg, const uint16_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_u64index_s64
  // CHECK: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u64, index_s64, )(pg, base, indices);
}

svuint32_t test_svld1uh_gather_u32index_u32(svbool_t pg, const uint16_t *base, svuint32_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_u32index_u32
  // CHECK: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.uxtw.index.nxv4i16(<vscale x 4 x i1> %[[PG]], i16* %base, <vscale x 4 x i32> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u32, index_u32, )(pg, base, indices);
}

svuint64_t test_svld1uh_gather_u64index_u64(svbool_t pg, const uint16_t *base, svuint64_t indices) {
  // CHECK-LABEL: test_svld1uh_gather_u64index_u64
  // CHECK: %[[PG]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %[[PG]], i16* %base, <vscale x 2 x i64> %indices)
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather_, u64, index_u64, )(pg, base, indices);
}

svint32_t test_svld1uh_gather_u32base_index_s32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svld1uh_gather_u32base_index_s32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 1
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %[[SHL]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u32base, _index_s32, )(pg, bases, index);
}

svint64_t test_svld1uh_gather_u64base_index_s64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svld1uh_gather_u64base_index_s64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 1
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u64base, _index_s64, )(pg, bases, index);
}

svuint32_t test_svld1uh_gather_u32base_index_u32(svbool_t pg, svuint32_t bases, int64_t index) {
  // CHECK-LABEL: test_svld1uh_gather_u32base_index_u32
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 1
  // CHECK: %[[LOAD:.*]] = call <vscale x 4 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %[[PG]], <vscale x 4 x i32> %bases, i64 %[[SHL]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
  // CHECK: ret <vscale x 4 x i32> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u32base, _index_u32, )(pg, bases, index);
}

svuint64_t test_svld1uh_gather_u64base_index_u64(svbool_t pg, svuint64_t bases, int64_t index) {
  // CHECK-LABEL: test_svld1uh_gather_u64base_index_u64
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[SHL:.*]] = shl i64 %index, 1
  // CHECK: %[[LOAD:.*]] = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %[[PG]], <vscale x 2 x i64> %bases, i64 %[[SHL]])
  // CHECK: %[[ZEXT:.*]] = zext <vscale x 2 x i16> %[[LOAD]] to <vscale x 2 x i64>
  // CHECK: ret <vscale x 2 x i64> %[[ZEXT]]
  return SVE_ACLE_FUNC(svld1uh_gather, _u64base, _index_u64, )(pg, bases, index);
}
