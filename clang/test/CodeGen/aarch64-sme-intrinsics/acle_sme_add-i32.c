// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

#ifdef SME_OVERLOADED_FORMS
#define SME_ACLE_FUNC(A1,A2_UNUSED,A3) A1##A3
#else
#define SME_ACLE_FUNC(A1,A2,A3) A1##A2##A3
#endif

// CHECK-C-LABEL: @test_svaddha_za32_u32(
// CHECK-CXX-LABEL: @_Z21test_svaddha_za32_u32u10__SVBool_tu10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv4i32(i32 0, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za32_u32(svbool_t pn, svbool_t pm, svuint32_t zn) {
  SME_ACLE_FUNC(svaddha_za32, _u32, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddha_za32_u32_1(
// CHECK-CXX-LABEL: @_Z23test_svaddha_za32_u32_1u10__SVBool_tu10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv4i32(i32 3, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za32_u32_1(svbool_t pn, svbool_t pm, svuint32_t zn) {
  SME_ACLE_FUNC(svaddha_za32, _u32, _m)(3, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddha_za32_s32(
// CHECK-CXX-LABEL: @_Z21test_svaddha_za32_s32u10__SVBool_tu10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv4i32(i32 0, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za32_s32(svbool_t pn, svbool_t pm, svint32_t zn) {
  SME_ACLE_FUNC(svaddha_za32, _s32, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddha_za32_s32_1(
// CHECK-CXX-LABEL: @_Z23test_svaddha_za32_s32_1u10__SVBool_tu10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv4i32(i32 3, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za32_s32_1(svbool_t pn, svbool_t pm, svint32_t zn) {
  SME_ACLE_FUNC(svaddha_za32, _s32, _m)(3, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za32_u32(
// CHECK-CXX-LABEL: @_Z21test_svaddva_za32_u32u10__SVBool_tu10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv4i32(i32 0, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za32_u32(svbool_t pn, svbool_t pm, svuint32_t zn) {
  SME_ACLE_FUNC(svaddva_za32, _u32, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za32_u32_1(
// CHECK-CXX-LABEL: @_Z23test_svaddva_za32_u32_1u10__SVBool_tu10__SVBool_tu12__SVUint32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv4i32(i32 3, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za32_u32_1(svbool_t pn, svbool_t pm, svuint32_t zn) {
  SME_ACLE_FUNC(svaddva_za32, _u32, _m)(3, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za32_s32(
// CHECK-CXX-LABEL: @_Z21test_svaddva_za32_s32u10__SVBool_tu10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv4i32(i32 0, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za32_s32(svbool_t pn, svbool_t pm, svint32_t zn) {
  SME_ACLE_FUNC(svaddva_za32, _s32, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za32_s32_1(
// CHECK-CXX-LABEL: @_Z23test_svaddva_za32_s32_1u10__SVBool_tu10__SVBool_tu11__SVInt32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv4i32(i32 3, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x i32> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za32_s32_1(svbool_t pn, svbool_t pm, svint32_t zn) {
  SME_ACLE_FUNC(svaddva_za32, _s32, _m)(3, pn, pm, zn);
}
