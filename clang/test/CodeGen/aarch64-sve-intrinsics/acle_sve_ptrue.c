// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

svbool_t test_svptrue_b8()
{
  // CHECK-LABEL: test_svptrue_b8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_b8();
}

svbool_t test_svptrue_b16()
{
  // CHECK-LABEL: test_svptrue_b16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svptrue_b16();
}

svbool_t test_svptrue_b32()
{
  // CHECK-LABEL: test_svptrue_b32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svptrue_b32();
}

svbool_t test_svptrue_b64()
{
  // CHECK-LABEL: test_svptrue_b64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svptrue_b64();
}

svbool_t test_svptrue_pat_b8()
{
  // CHECK-LABEL: test_svptrue_pat_b8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 0)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_POW2);
}

svbool_t test_svptrue_pat_b8_1()
{
  // CHECK-LABEL: test_svptrue_pat_b8_1
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 1)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL1);
}

svbool_t test_svptrue_pat_b8_2()
{
  // CHECK-LABEL: test_svptrue_pat_b8_2
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL2);
}

svbool_t test_svptrue_pat_b8_3()
{
  // CHECK-LABEL: test_svptrue_pat_b8_3
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 3)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL3);
}

svbool_t test_svptrue_pat_b8_4()
{
  // CHECK-LABEL: test_svptrue_pat_b8_4
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 4)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL4);
}

svbool_t test_svptrue_pat_b8_5()
{
  // CHECK-LABEL: test_svptrue_pat_b8_5
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 5)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL5);
}

svbool_t test_svptrue_pat_b8_6()
{
  // CHECK-LABEL: test_svptrue_pat_b8_6
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 6)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL6);
}

svbool_t test_svptrue_pat_b8_7()
{
  // CHECK-LABEL: test_svptrue_pat_b8_7
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 7)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL7);
}

svbool_t test_svptrue_pat_b8_8()
{
  // CHECK-LABEL: test_svptrue_pat_b8_8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 8)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL8);
}

svbool_t test_svptrue_pat_b8_9()
{
  // CHECK-LABEL: test_svptrue_pat_b8_9
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 9)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL16);
}

svbool_t test_svptrue_pat_b8_10()
{
  // CHECK-LABEL: test_svptrue_pat_b8_10
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 10)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL32);
}

svbool_t test_svptrue_pat_b8_11()
{
  // CHECK-LABEL: test_svptrue_pat_b8_11
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 11)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL64);
}

svbool_t test_svptrue_pat_b8_12()
{
  // CHECK-LABEL: test_svptrue_pat_b8_12
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 12)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL128);
}

svbool_t test_svptrue_pat_b8_13()
{
  // CHECK-LABEL: test_svptrue_pat_b8_13
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 13)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_VL256);
}

svbool_t test_svptrue_pat_b8_14()
{
  // CHECK-LABEL: test_svptrue_pat_b8_14
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 29)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_MUL4);
}

svbool_t test_svptrue_pat_b8_15()
{
  // CHECK-LABEL: test_svptrue_pat_b8_15
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 30)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_MUL3);
}

svbool_t test_svptrue_pat_b8_16()
{
  // CHECK-LABEL: test_svptrue_pat_b8_16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  return svptrue_pat_b8(SV_ALL);
}

svbool_t test_svptrue_pat_b16()
{
  // CHECK-LABEL: test_svptrue_pat_b16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 0)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svptrue_pat_b16(SV_POW2);
}

svbool_t test_svptrue_pat_b32()
{
  // CHECK-LABEL: test_svptrue_pat_b32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 1)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svptrue_pat_b32(SV_VL1);
}

svbool_t test_svptrue_pat_b64()
{
  // CHECK-LABEL: test_svptrue_pat_b64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 2)
  // CHECK: %[[CAST:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[CAST]]
  return svptrue_pat_b64(SV_VL2);
}
