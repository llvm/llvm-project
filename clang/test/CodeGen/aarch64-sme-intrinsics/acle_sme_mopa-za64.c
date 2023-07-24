// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme-f64f64 -target-feature +sme-i16i64 -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme-f64f64 -target-feature +sme-i16i64 -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme-f64f64 -target-feature +sme-i16i64 -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme-f64f64 -target-feature +sme-i16i64 -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme-f64f64 -target-feature +sme-i16i64 -target-feature +sve -target-feature +bf16 -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

#ifdef SME_OVERLOADED_FORMS
#define SME_ACLE_FUNC(A1,A2_UNUSED,A3) A1##A3
#else
#define SME_ACLE_FUNC(A1,A2,A3) A1##A2##A3
#endif

// CHECK-C-LABEL: @test_svmopa_za64_s16(
// CHECK-CXX-LABEL: @_Z20test_svmopa_za64_s16u10__SVBool_tu10__SVBool_tu11__SVInt16_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.smopa.wide.nxv8i16(i32 1, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i1> [[TMP1]], <vscale x 8 x i16> [[ZN:%.*]], <vscale x 8 x i16> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za64_s16(svbool_t pn, svbool_t pm, svint16_t zn, svint16_t zm) {
  SME_ACLE_FUNC(svmopa_za64, _s16, _m)(1, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svmopa_za64_u16(
// CHECK-CXX-LABEL: @_Z20test_svmopa_za64_u16u10__SVBool_tu10__SVBool_tu12__SVUint16_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.umopa.wide.nxv8i16(i32 0, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i1> [[TMP1]], <vscale x 8 x i16> [[ZN:%.*]], <vscale x 8 x i16> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za64_u16(svbool_t pn, svbool_t pm, svuint16_t zn, svuint16_t zm) {
  SME_ACLE_FUNC(svmopa_za64, _u16, _m)(0, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svmopa_za64_f64(
// CHECK-CXX-LABEL: @_Z20test_svmopa_za64_f64u10__SVBool_tu10__SVBool_tu13__SVFloat64_tu13__SVFloat64_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.mopa.nxv2f64(i32 1, <vscale x 2 x i1> [[TMP0]], <vscale x 2 x i1> [[TMP1]], <vscale x 2 x double> [[ZN:%.*]], <vscale x 2 x double> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za64_f64(svbool_t pn, svbool_t pm, svfloat64_t zn, svfloat64_t zm) {
  SME_ACLE_FUNC(svmopa_za64, _f64, _m)(1, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svsumopa_za64_s16(
// CHECK-CXX-LABEL: @_Z22test_svsumopa_za64_s16u10__SVBool_tu10__SVBool_tu11__SVInt16_tu12__SVUint16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.sumopa.wide.nxv8i16(i32 0, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i1> [[TMP1]], <vscale x 8 x i16> [[ZN:%.*]], <vscale x 8 x i16> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svsumopa_za64_s16(svbool_t pn, svbool_t pm, svint16_t zn, svuint16_t zm) {
 SME_ACLE_FUNC(svsumopa_za64, _s16, _m)(0, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svusmopa_za64_u16(
// CHECK-CXX-LABEL: @_Z22test_svusmopa_za64_u16u10__SVBool_tu10__SVBool_tu12__SVUint16_tu11__SVInt16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.usmopa.wide.nxv8i16(i32 2, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i1> [[TMP1]], <vscale x 8 x i16> [[ZN:%.*]], <vscale x 8 x i16> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svusmopa_za64_u16(svbool_t pn, svbool_t pm, svuint16_t zn, svint16_t zm) {
  SME_ACLE_FUNC(svusmopa_za64, _u16, _m)(2, pn, pm, zn, zm);
}
