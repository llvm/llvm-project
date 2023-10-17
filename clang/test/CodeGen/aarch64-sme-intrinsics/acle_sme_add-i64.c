// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme-i16i64 -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme-i16i64 -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme-i16i64 -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme-i16i64 -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme-i16i64 -target-feature +sve -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

#ifdef SME_OVERLOADED_FORMS
#define SME_ACLE_FUNC(A1,A2_UNUSED,A3) A1##A3
#else
#define SME_ACLE_FUNC(A1,A2,A3) A1##A2##A3
#endif

// CHECK-C-LABEL: @test_svaddha_za64_u64(
// CHECK-CXX-LABEL: @_Z21test_svaddha_za64_u64u10__SVBool_tu10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv2i64(i32 0, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za64_u64(svbool_t pn, svbool_t pm, svuint64_t zn) {
  SME_ACLE_FUNC(svaddha_za64, _u64, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddha_za64_u64_1(
// CHECK-CXX-LABEL: @_Z23test_svaddha_za64_u64_1u10__SVBool_tu10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv2i64(i32 7, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za64_u64_1(svbool_t pn, svbool_t pm, svuint64_t zn) {
  SME_ACLE_FUNC(svaddha_za64, _u64, _m)(7, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddha_za64_s64(
// CHECK-CXX-LABEL: @_Z21test_svaddha_za64_s64u10__SVBool_tu10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv2i64(i32 0, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za64_s64(svbool_t pn, svbool_t pm, svint64_t zn) {
  SME_ACLE_FUNC(svaddha_za64, _s64, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddha_za64_s64_1(
// CHECK-CXX-LABEL: @_Z23test_svaddha_za64_s64_1u10__SVBool_tu10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addha.nxv2i64(i32 7, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddha_za64_s64_1(svbool_t pn, svbool_t pm, svint64_t zn) {
  SME_ACLE_FUNC(svaddha_za64, _s64, _m)(7, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za64_u64(
// CHECK-CXX-LABEL: @_Z21test_svaddva_za64_u64u10__SVBool_tu10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv2i64(i32 0, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za64_u64(svbool_t pn, svbool_t pm, svuint64_t zn) {
  SME_ACLE_FUNC(svaddva_za64, _u64, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za64_u64_1(
// CHECK-CXX-LABEL: @_Z23test_svaddva_za64_u64_1u10__SVBool_tu10__SVBool_tu12__SVUint64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv2i64(i32 7, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za64_u64_1(svbool_t pn, svbool_t pm, svuint64_t zn) {
  SME_ACLE_FUNC(svaddva_za64, _u64, _m)(7, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za64_s64(
// CHECK-CXX-LABEL: @_Z21test_svaddva_za64_s64u10__SVBool_tu10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv2i64(i32 0, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za64_s64(svbool_t pn, svbool_t pm, svint64_t zn) {
  SME_ACLE_FUNC(svaddva_za64, _s64, _m)(0, pn, pm, zn);
}

// CHECK-C-LABEL: @test_svaddva_za64_s64_1(
// CHECK-CXX-LABEL: @_Z23test_svaddva_za64_s64_1u10__SVBool_tu10__SVBool_tu11__SVInt64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.addva.nxv2i64(i32 7, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x i64> [[ZN:%.*]])
// CHECK-NEXT:    ret void
//
void test_svaddva_za64_s64_1(svbool_t pn, svbool_t pm, svint64_t zn) {
  SME_ACLE_FUNC(svaddva_za64, _s64, _m)(7, pn, pm, zn);
}
