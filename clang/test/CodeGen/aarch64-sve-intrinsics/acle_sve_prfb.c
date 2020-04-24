// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

void test_svprfb(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 0)
  return svprfb(pg, base, SV_PLDL1KEEP);
}

void test_svprfb_1(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_1
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 1)
  return svprfb(pg, base, SV_PLDL1STRM);
}

void test_svprfb_2(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_2
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 2)
  return svprfb(pg, base, SV_PLDL2KEEP);
}

void test_svprfb_3(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_3
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 3)
  return svprfb(pg, base, SV_PLDL2STRM);
}

void test_svprfb_4(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_4
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 4)
  return svprfb(pg, base, SV_PLDL3KEEP);
}

void test_svprfb_5(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_5
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 5)
  return svprfb(pg, base, SV_PLDL3STRM);
}

void test_svprfb_6(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_6
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 8)
  return svprfb(pg, base, SV_PSTL1KEEP);
}

void test_svprfb_7(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_7
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 9)
  return svprfb(pg, base, SV_PSTL1STRM);
}

void test_svprfb_8(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_8
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 10)
  return svprfb(pg, base, SV_PSTL2KEEP);
}

void test_svprfb_9(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_9
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 11)
  return svprfb(pg, base, SV_PSTL2STRM);
}

void test_svprfb_10(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_10
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 12)
  return svprfb(pg, base, SV_PSTL3KEEP);
}

void test_svprfb_11(svbool_t pg, const void *base)
{
  // CHECK-LABEL: test_svprfb_11
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %base, i32 13)
  return svprfb(pg, base, SV_PSTL3STRM);
}

void test_svprfb_vnum(svbool_t pg, const void *base, int64_t vnum)
{
  // CHECK-LABEL: test_svprfb_vnum
  // CHECK: %[[BASE:.*]] = bitcast i8* %base to <vscale x 16 x i8>*
  // CHECK: %[[GEP:.*]] = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: @llvm.aarch64.sve.prf.nxv16i1(<vscale x 16 x i1> %pg, i8* %[[GEP]], i32 0)
  return svprfb_vnum(pg, base, vnum, SV_PLDL1KEEP);
}
