// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -DDISABLE_SME_ATTRIBUTES -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -DDISABLE_SME_ATTRIBUTES -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -DDISABLE_SME_ATTRIBUTES -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

#ifdef DISABLE_SME_ATTRIBUTES
#define ARM_STREAMING_ATTR
#else
#define ARM_STREAMING_ATTR __attribute__((arm_streaming))
#endif

// CHECK-C-LABEL:   @test_svst1_hor_za8(
// CHECK-CXX-LABEL: @_Z18test_svst1_hor_za8ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1b.horiz(<vscale x 16 x i1> [[PG:%.*]], [[PTRTY:ptr|i8\*]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 15
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1b.horiz(<vscale x 16 x i1> [[PG]], [[PTRTY]] [[PTR]], i32 0, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_hor_za8(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_hor_za8(0, slice_base, 0, pg, ptr);
  svst1_hor_za8(0, slice_base, 15, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_hor_za16(
// CHECK-CXX-LABEL: @_Z19test_svst1_hor_za16ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1h.horiz(<vscale x 8 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 7
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1h.horiz(<vscale x 8 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 1, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_hor_za16(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_hor_za16(0, slice_base, 0, pg, ptr);
  svst1_hor_za16(1, slice_base, 7, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_hor_za32(
// CHECK-CXX-LABEL: @_Z19test_svst1_hor_za32ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 3
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 3, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_hor_za32(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_hor_za32(0, slice_base, 0, pg, ptr);
  svst1_hor_za32(3, slice_base, 3, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_hor_za64(
// CHECK-CXX-LABEL: @_Z19test_svst1_hor_za64ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1d.horiz(<vscale x 2 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 1
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1d.horiz(<vscale x 2 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 7, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_hor_za64(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_hor_za64(0, slice_base, 0, pg, ptr);
  svst1_hor_za64(7, slice_base, 1, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_hor_za128(
// CHECK-CXX-LABEL: @_Z20test_svst1_hor_za128ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 1 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv1i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1q.horiz(<vscale x 1 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1q.horiz(<vscale x 1 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 15, i32 [[SLICE_BASE]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_hor_za128(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_hor_za128(0, slice_base, 0, pg, ptr);
  svst1_hor_za128(15, slice_base, 0, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_ver_za8(
// CHECK-CXX-LABEL: @_Z18test_svst1_ver_za8ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1b.vert(<vscale x 16 x i1> [[PG:%.*]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 15
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1b.vert(<vscale x 16 x i1> [[PG]], [[PTRTY]] [[PTR]], i32 0, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_ver_za8(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_ver_za8(0, slice_base, 0, pg, ptr);
  svst1_ver_za8(0, slice_base, 15, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_ver_za16(
// CHECK-CXX-LABEL: @_Z19test_svst1_ver_za16ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1h.vert(<vscale x 8 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 7
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1h.vert(<vscale x 8 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 1, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_ver_za16(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_ver_za16(0, slice_base, 0, pg, ptr);
  svst1_ver_za16(1, slice_base, 7, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_ver_za32(
// CHECK-CXX-LABEL: @_Z19test_svst1_ver_za32ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1w.vert(<vscale x 4 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 3
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1w.vert(<vscale x 4 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 3, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_ver_za32(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_ver_za32(0, slice_base, 0, pg, ptr);
  svst1_ver_za32(3, slice_base, 3, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_ver_za64(
// CHECK-CXX-LABEL: @_Z19test_svst1_ver_za64ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1d.vert(<vscale x 2 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        [[TILESLICE1:%.*]] = add i32 [[SLICE_BASE]], 1
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1d.vert(<vscale x 2 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 7, i32 [[TILESLICE1]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_ver_za64(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_ver_za64(0, slice_base, 0, pg, ptr);
  svst1_ver_za64(7, slice_base, 1, pg, ptr);
}

// CHECK-C-LABEL:   @test_svst1_ver_za128(
// CHECK-CXX-LABEL: @_Z20test_svst1_ver_za128ju10__SVBool_tPv(
// CHECK-NEXT:      entry:
// CHECK-NEXT:        [[TMP0:%.*]] = tail call <vscale x 1 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv1i1(<vscale x 16 x i1> [[PG:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1q.vert(<vscale x 1 x i1> [[TMP0]], [[PTRTY]] [[PTR:%.*]], i32 0, i32 [[SLICE_BASE:%.*]])
// CHECK-NEXT:        tail call void @llvm.aarch64.sme.st1q.vert(<vscale x 1 x i1> [[TMP0]], [[PTRTY]] [[PTR]], i32 15, i32 [[SLICE_BASE]])
// CHECK-NEXT:        ret void
//
ARM_STREAMING_ATTR void test_svst1_ver_za128(uint32_t slice_base, svbool_t pg, void *ptr) {
  svst1_ver_za128(0, slice_base, 0, pg, ptr);
  svst1_ver_za128(15, slice_base, 0, pg, ptr);
}
